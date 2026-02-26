#include "phase5/runtime/ops/runtime_ops.h"

#include "phase3/evaluator_parts/internal_helpers.h"

namespace spark {

// Keep Interpreter entry-points thin; container arithmetic lives in phase5/runtime/ops/*.
double Interpreter::to_number(const Value& value) {
  return runtime_ops::to_number(value);
}

bool Interpreter::truthy(const Value& value) {
  switch (value.kind) {
    case Value::Kind::Nil:
      return false;
    case Value::Kind::Bool:
      return value.bool_value;
    case Value::Kind::Int:
      return value.int_value != 0;
    case Value::Kind::Double:
      return value.double_value != 0.0;
    case Value::Kind::String:
      return !value.string_value.empty();
    case Value::Kind::Numeric:
      return !numeric_value_is_zero(value);
    case Value::Kind::List:
      if (!value.list_value.empty()) {
        return true;
      }
      if (value.list_cache.materialized_version == value.list_cache.version &&
          !value.list_cache.promoted_f64.empty()) {
        return true;
      }
      return false;
    default:
      return true;
  }
}

Value Interpreter::eval_unary(UnaryOp op, const Value& operand) const {
  return runtime_ops::controllers::eval_unary_orchestrator(op, operand);
}

Value Interpreter::eval_binary(BinaryOp op, const Value& left, const Value& right) const {
  return runtime_ops::controllers::eval_binary_orchestrator(op, left, right);
}

}  // namespace spark
