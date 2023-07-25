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
 * @brief Benchmark cornerstone octree generation on the GPU
 *
 * @author Sebastian Keller <sebastian.f.keller@gmail.com>
 */

#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "util/random.hpp"
#include "util/timing.cuh"
#include "sfc/bitops.hpp"
#include "tree/csarray_gpu.cuh"
#include "tree/octree_gpu.cuh"

using namespace cstone;

int main()
{
    using KeyType = uint64_t;
    Box<double> box{-1, 1};

    unsigned numParticles = 2000000;
    unsigned bucketSize   = 16;

    RandomCoordinates<double, KeyType> randomBox(numParticles, box);

    thrust::device_vector<KeyType>  octree = std::vector<KeyType>{0, nodeRange<KeyType>(0)};
    thrust::device_vector<unsigned> counts = std::vector<unsigned>{numParticles};

    thrust::device_vector<KeyType>       tmpTree;
    thrust::device_vector<TreeNodeIndex> workArray;

    thrust::device_vector<KeyType> particleKeys(randomBox.keys().begin(), randomBox.keys().end());

    auto fullBuild = [&]()
    {
        while (!updateOctreeGpu(rawPtr(particleKeys), rawPtr(particleKeys) + numParticles, bucketSize, octree, counts,
                                tmpTree, workArray))
            ;
    };

    float buildTime = timeGpu(fullBuild);
    std::cout << "build time from scratch " << buildTime / 1000 << " nNodes(tree): " << nNodes(octree)
              << " particle count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    auto updateTree = [&]()
    {
        updateOctreeGpu(rawPtr(particleKeys), rawPtr(particleKeys) + numParticles, bucketSize, octree, counts, tmpTree,
                        workArray);
    };

    float updateTime = timeGpu(updateTree);
    std::cout << "build time with guess " << updateTime / 1000 << " nNodes(tree): " << nNodes(octree)
              << " particle count: " << thrust::reduce(counts.begin(), counts.end(), 0) << std::endl;

    OctreeData<KeyType, GpuTag> linkedOctree;
    linkedOctree.resize(nNodes(octree));
    auto buildInternal = [&]() { buildLinkedTreeGpu(rawPtr(octree), linkedOctree.data()); };

    float internalBuildTime = timeGpu(buildInternal);
    std::cout << "linked octree build time " << internalBuildTime / 1000 << std::endl << std::endl;

    thrust::host_vector<TreeNodeIndex> levelRange = linkedOctree.levelRange; // download from GPU
    for (int i = 0; i < levelRange.size(); ++i)
    {
        int numNodes = levelRange[i + 1] - levelRange[i];
        if (numNodes == 0) { break; }
        std::cout << "number of nodes at level " << i << ": " << numNodes << std::endl;
    }
}
