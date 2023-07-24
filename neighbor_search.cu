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
 * @brief  Find neighbors in Space-Filling-Curve sorted x,y,z arrays
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>
#include <thrust/device_vector.h>

#include "util/cuda_utils.hpp"
#include "util/random.hpp"
#include "util/timing.cuh"

#include "findneighbors.hpp"
#include "findneighbors_warps.cuh"

// uncomment to enable warp-level optimized neighbor search
// #define USE_WARPS

using namespace cstone;

template<class T, class KeyType>
__global__ void findNeighborsKernel(const T* x, const T* y, const T* z, const T* h, LocalIndex firstId,
                                    LocalIndex lastId, const Box<T> box, const OctreeNsView<T, KeyType> treeView,
                                    unsigned ngmax, LocalIndex* neighbors, unsigned* neighborsCount)
{
    cstone::LocalIndex tid = blockDim.x * blockIdx.x + threadIdx.x;
    cstone::LocalIndex id  = firstId + tid;
    if (id >= lastId) { return; }

    neighborsCount[id] = findNeighbors(id, x, y, z, h, treeView, box, ngmax, neighbors + tid * ngmax);
}

template<class T, class KeyType>
void benchmarkGpu()
{
    Box<T> box{0, 1, BoundaryType::open};
    int    numParticles = 2000000;

    RandomCoordinates<T, KeyType> coords(numParticles, box);
    std::vector<T>                h(numParticles, 0.012);

    int maxNeighbors = 200;

    std::vector<LocalIndex> neighborsCPU(maxNeighbors * numParticles);
    std::vector<unsigned>   neighborsCountCPU(numParticles);

    const T*    x    = coords.x().data();
    const T*    y    = coords.y().data();
    const T*    z    = coords.z().data();
    const auto* keys = (KeyType*)(coords.particleKeys().data());

    unsigned bucketSize   = 64; // maximum number of particles per leaf node
    auto [csTree, counts] = computeOctree(keys, keys + numParticles, bucketSize);
    OctreeData<KeyType, CpuTag> octree;
    octree.resize(nNodes(csTree));
    updateInternalTree<KeyType>(csTree.data(), octree.data());
    const TreeNodeIndex* childOffsets = octree.childOffsets.data();
    const TreeNodeIndex* toLeafOrder  = octree.internalToLeaf.data();

    std::vector<LocalIndex> layout(nNodes(csTree) + 1);
    std::exclusive_scan(counts.begin(), counts.end() + 1, layout.begin(), 0);

    std::vector<Vec3<T>> nodeCenters(octree.numNodes), nodeSizes(octree.numNodes);
    nodeFpCenters(octree.prefixes.data(), octree.numNodes, nodeCenters.data(), nodeSizes.data(), box);

    OctreeNsView<T, KeyType> nsView{nodeCenters.data(), nodeSizes.data(), octree.childOffsets.data(),
                                    octree.internalToLeaf.data(), layout.data()};

    auto findNeighborsCpu = [&]()
    {
#pragma omp parallel for
        for (LocalIndex i = 0; i < numParticles; ++i)
        {
            neighborsCountCPU[i] =
                findNeighbors(i, x, y, z, h.data(), nsView, box, maxNeighbors, neighborsCPU.data() + i * maxNeighbors);
        }
    };

    float cpuTime = timeCpu(findNeighborsCpu);

    std::cout << "CPU time " << cpuTime << " s" << std::endl;
    std::copy(neighborsCountCPU.data(), neighborsCountCPU.data() + 64, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    std::vector<cstone::LocalIndex> neighborsGPU(maxNeighbors * numParticles);
    std::vector<unsigned>           neighborsCountGPU(numParticles);

    thrust::device_vector<T> d_x(coords.x().begin(), coords.x().end());
    thrust::device_vector<T> d_y(coords.y().begin(), coords.y().end());
    thrust::device_vector<T> d_z(coords.z().begin(), coords.z().end());
    thrust::device_vector<T> d_h = h;

    thrust::device_vector<Vec3<T>>       d_nodeCenters    = nodeCenters;
    thrust::device_vector<Vec3<T>>       d_nodeSizes      = nodeSizes;
    thrust::device_vector<TreeNodeIndex> d_childOffsets   = octree.childOffsets;
    thrust::device_vector<TreeNodeIndex> d_internalToLeaf = octree.internalToLeaf;
    thrust::device_vector<LocalIndex>    d_layout         = layout;

    OctreeNsView<T, KeyType> nsViewGpu{rawPtr(d_nodeCenters), rawPtr(d_nodeSizes), rawPtr(d_childOffsets),
                                       rawPtr(d_internalToLeaf), rawPtr(d_layout)};

    thrust::device_vector<LocalIndex> d_neighbors(neighborsGPU.size());
    thrust::device_vector<unsigned>   d_neighborsCount(neighborsCountGPU.size());

    auto findNeighborsLambda = [&]()
    {
#ifdef USE_WARPS
        // the fast warp-aware version
        findNeighborsBT(0, n, rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), nsViewGpu, box,
                        rawPtr(d_neighborsCount), rawPtr(d_neighbors), maxNeighbors);
#else
        findNeighborsKernel<<<iceil(numParticles, 128), 128>>>(rawPtr(d_x), rawPtr(d_y), rawPtr(d_z), rawPtr(d_h), 0,
                                                               numParticles, box, nsViewGpu, maxNeighbors,
                                                               rawPtr(d_neighbors), rawPtr(d_neighborsCount));
#endif
    };

    float gpuTime = timeGpu(findNeighborsLambda);

    thrust::copy(d_neighborsCount.begin(), d_neighborsCount.end(), neighborsCountGPU.begin());
    thrust::copy(d_neighbors.begin(), d_neighbors.end(), neighborsGPU.begin());

    std::cout << "GPU time " << gpuTime / 1000 << " s" << std::endl;
    std::copy(neighborsCountGPU.data(), neighborsCountGPU.data() + 64, std::ostream_iterator<int>(std::cout, " "));
    std::cout << std::endl;

    int numFails     = 0;
    int numFailsList = 0;
    for (int i = 0; i < numParticles; ++i)
    {
        std::sort(neighborsCPU.data() + i * maxNeighbors,
                  neighborsCPU.data() + i * maxNeighbors + neighborsCountCPU[i]);

        std::vector<cstone::LocalIndex> nilist(neighborsCountGPU[i]);
        for (unsigned j = 0; j < neighborsCountGPU[i]; ++j)
        {
#ifdef USE_WARPS
            // access pattern for the warp-aware version
            size_t warpOffset = (i / TravConfig::targetSize) * TravConfig::targetSize * maxNeighbors;
            size_t laneOffset = i % TravConfig::targetSize;
            nilist[j]         = neighborsGPU[warpOffset + TravConfig::targetSize * j + laneOffset];
            nilist[j]         = neighborsGPU[warpOffset + TravConfig::targetSize * j + laneOffset];
#else
            nilist[j] = neighborsGPU[i * maxNeighbors + j];
#endif
        }
        std::sort(nilist.begin(), nilist.end());

        if (neighborsCountGPU[i] != neighborsCountCPU[i])
        {
            std::cout << i << " " << neighborsCountGPU[i] << " " << neighborsCountCPU[i] << std::endl;
            numFails++;
        }

        if (!std::equal(begin(nilist), end(nilist), neighborsCPU.begin() + i * maxNeighbors)) { numFailsList++; }
    }

    bool allEqual = std::equal(begin(neighborsCountGPU), end(neighborsCountGPU), begin(neighborsCountCPU));
    if (allEqual)
        std::cout << "Neighbor counts: PASS\n";
    else
        std::cout << "Neighbor counts: FAIL " << numFails << std::endl;

    std::cout << "numFailsList " << numFailsList << std::endl;
}

int main() { benchmarkGpu<double, uint64_t>(); }
