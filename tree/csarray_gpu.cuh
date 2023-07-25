/*
 * MIT License
 *
 * Copyright (c) 2021 CSCS, ETH Zurich
 *               2021 University of Basel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*! @file
 * @brief Generation of local and global octrees in cornerstone format on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 *
 * See octree.hpp for a description of the cornerstone format.
 */

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include "../util/cuda_utils.hpp"
#include "csarray.hpp"

namespace cstone
{

//! @brief see computeNodeCounts
template<class KeyType>
__global__ void computeNodeCountsKernel(const KeyType* tree,
                                        unsigned* counts,
                                        TreeNodeIndex nNodes,
                                        const KeyType* codesStart,
                                        const KeyType* codesEnd,
                                        unsigned maxCount)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nNodes) { counts[tid] = calculateNodeCount(tree[tid], tree[tid + 1], codesStart, codesEnd, maxCount); }
}

template<class KeyType>
void computeNodeCountsGpu(const KeyType* tree,
                          unsigned* counts,
                          TreeNodeIndex numNodes,
                          const KeyType* firstKey,
                          const KeyType* lastKey,
                          unsigned maxCount)
{
    constexpr unsigned numThreads = 256;
    computeNodeCountsKernel<<<iceil(numNodes, numThreads), numThreads>>>(tree, counts, numNodes, firstKey, lastKey,
                                                                         maxCount);
}

//! @brief this symbol is used to keep track of octree structure changes and detect convergence
__device__ int rebalanceChangeCounter;

__global__ void resetRebalanceCounter() { rebalanceChangeCounter = 0; }

/*! @brief Compute split or fuse decision for each octree node in parallel
 *
 * @tparam KeyType         32- or 64-bit unsigned integer type
 * @param[in] tree         octree nodes given as Morton codes of length @a numNodes
 *                         needs to satisfy the octree invariants
 * @param[in] counts       output particle counts per node, length = @a numNodes
 * @param[in] numNodes     number of nodes in tree
 * @param[in] bucketSize   maximum particle count per (leaf) node and
 *                         minimum particle count (strictly >) for (implicit) internal nodes
 * @param[out] nodeOps     stores rebalance decision result for each node, length = @a numNodes
 * @param[out] converged   stores 0 upon return if converged, a non-zero positive integer otherwise.
 *                         The storage location is accessed concurrently and cuda-memcheck might detect
 *                         a data race, but this is irrelevant for correctness.
 *
 * For each node i in the tree, in nodeOps[i], stores
 *  - 0 if to be merged
 *  - 1 if unchanged,
 *  - 8 if to be split.
 */
template<class KeyType>
__global__ void rebalanceDecisionKernel(
    const KeyType* tree, const unsigned* counts, TreeNodeIndex numNodes, unsigned bucketSize, TreeNodeIndex* nodeOps)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numNodes)
    {
        int decision = calculateNodeOp(tree, tid, counts, bucketSize);
        if (decision != 1) { rebalanceChangeCounter = 1; }
        nodeOps[tid] = decision;
    }
}

template<class KeyType>
TreeNodeIndex computeNodeOpsGpu(
    const KeyType* tree, TreeNodeIndex numNodes, const unsigned* counts, unsigned bucketSize, TreeNodeIndex* nodeOps)
{
    resetRebalanceCounter<<<1, 1>>>();

    constexpr unsigned nThreads = 512;
    rebalanceDecisionKernel<<<iceil(numNodes, nThreads), nThreads>>>(tree, counts, numNodes, bucketSize, nodeOps);

    size_t nodeOpsSize = numNodes + 1;
    thrust::exclusive_scan(thrust::device, nodeOps, nodeOps + nodeOpsSize, nodeOps);

    TreeNodeIndex newNumNodes;
    thrust::copy_n(thrust::device_pointer_cast(nodeOps) + nodeOpsSize - 1, 1, &newNumNodes);

    return newNumNodes;
}

/*! @brief construct new nodes in the balanced tree
 *
 * @tparam KeyType         32- or 64-bit unsigned integer type
 * @param[in]  oldTree     old cornerstone octree, length = numOldNodes + 1
 * @param[in]  nodeOps     transformation codes for old tree, length = numOldNodes + 1
 * @param[in]  numOldNodes number of nodes in @a oldTree
 * @param[out] newTree     the rebalanced tree, length = nodeOps[numOldNodes] + 1
 */
template<class KeyType>
__global__ void
processNodes(const KeyType* oldTree, const TreeNodeIndex* nodeOps, TreeNodeIndex numOldNodes, KeyType* newTree)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < numOldNodes) { processNode(tid, oldTree, nodeOps, newTree); }
}

template<class KeyType>
bool rebalanceTreeGpu(const KeyType* tree,
                      TreeNodeIndex numNodes,
                      TreeNodeIndex newNumNodes,
                      const TreeNodeIndex* nodeOps,
                      KeyType* newTree)
{
    constexpr unsigned nThreads = 512;
    processNodes<<<iceil(numNodes, nThreads), nThreads>>>(tree, nodeOps, numNodes, newTree);
    thrust::fill_n(thrust::device_pointer_cast(newTree + newNumNodes), 1, nodeRange<KeyType>(0));

    int changeCounter;
    checkGpuErrors(cudaMemcpyFromSymbol(&changeCounter, rebalanceChangeCounter, sizeof(int)));

    return changeCounter == 0;
}

/*! @brief update the octree with a single rebalance/count step
 *
 * @tparam KeyType           32- or 64-bit unsigned integer for morton code
 * @param[in]    firstKey    first local particle SFC key
 * @param[in]    lastKey     last local particle SFC key
 * @param[in]    bucketSize  maximum number of particles per node
 * @param[inout] tree        the octree leaf nodes (cornerstone format)
 * @param[inout] counts      the octree leaf node particle count
 * @param[-]     tmpTree     temporary array, will be resized as needed
 * @param[-]     nodeOps     temporary array, will be resized as needed
 * @param[in]    maxCount    if actual node counts are higher, they will be capped to @p maxCount
 * @return                   true if converged, false otherwise
 */
template<class KeyType, class DevKeyVec, class DevCountVec, class DevIdxVec>
bool updateOctreeGpu(const KeyType* firstKey,
                     const KeyType* lastKey,
                     unsigned bucketSize,
                     DevKeyVec& tree,
                     DevCountVec& counts,
                     DevKeyVec& tmpTree,
                     DevIdxVec& nodeOps,
                     unsigned maxCount = std::numeric_limits<unsigned>::max())
{
    nodeOps.resize(tree.size());
    TreeNodeIndex newNumNodes =
        computeNodeOpsGpu(rawPtr(tree), nNodes(tree), rawPtr(counts), bucketSize, rawPtr(nodeOps));

    tmpTree.resize(newNumNodes + 1);
    bool converged = rebalanceTreeGpu(rawPtr(tree), nNodes(tree), newNumNodes, rawPtr(nodeOps), rawPtr(tmpTree));
    swap(tree, tmpTree);

    counts.resize(nNodes(tree));
    computeNodeCountsGpu(rawPtr(tree), rawPtr(counts), nNodes(tree), firstKey, lastKey, maxCount);

    return converged;
}

} // namespace cstone
