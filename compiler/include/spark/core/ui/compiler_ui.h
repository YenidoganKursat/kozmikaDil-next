#pragma once

#include "spark/core/behavior/compiler_behavior.h"
#include "spark/core/mode/compiler_mode.h"

namespace spark::core::ui {

class CompilerUiFacade {
 public:
  explicit CompilerUiFacade(behavior::CompilerBehavior& behavior)
      : behavior_(behavior) {}

  bool parse_and_typecheck(const dto::FilePath& source_file, dto::ProgramBundle& out_bundle) {
    return behavior_.parse_and_typecheck(source_file, out_bundle);
  }

  bool lower_to_products(const dto::ProgramBundle& bundle,
                         dto::PipelineProducts& out_products) {
    return behavior_.lower_to_products(bundle, out_products);
  }

 private:
  behavior::CompilerBehavior& behavior_;
};

}  // namespace spark::core::ui
