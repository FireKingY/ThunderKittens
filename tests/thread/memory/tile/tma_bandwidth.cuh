#pragma once

#include "testing_flags.cuh"

#ifdef TEST_THREAD_MEMORY_TILE_TMA_BW

#include "testing_commons.cuh"

namespace thread {
namespace memory {
namespace tile {
namespace tma_bandwidth {

void tests(test_data &results);

}
}
}
}

#endif
