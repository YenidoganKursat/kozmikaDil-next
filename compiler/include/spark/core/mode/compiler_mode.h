#pragma once

#include "spark/core/dto/compiler_pipeline_dto.h"

namespace spark::core::mode {

struct CompilerMode {
  bool allow_t5 = false;
  bool explain_layout = false;
  dto::BuildTuning tuning;
};

}  // namespace spark::core::mode
