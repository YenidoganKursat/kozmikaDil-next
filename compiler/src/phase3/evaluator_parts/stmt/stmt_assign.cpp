#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <limits>
#include <string>

#if defined(__APPLE__)
#include <mach/mach_time.h>
#endif

#include "../internal_helpers.h"

namespace spark {

namespace {

using I128 = __int128_t;
using U128 = __uint128_t;

I128 assign_i128_max() {
  return static_cast<I128>((~U128{0}) >> 1);
}

I128 assign_i128_min() {
  return -assign_i128_max() - 1;
}

bool assign_i128_positive_pow2(I128 value) {
  if (value <= 0) {
    return false;
  }
  const auto bits = static_cast<U128>(value);
  return (bits & (bits - 1U)) == 0U;
}

bool assign_try_fast_mod_pow2(I128 lhs, I128 rhs, I128& out) {
  if (!assign_i128_positive_pow2(rhs) || lhs < 0) {
    return false;
  }
  out = lhs & (rhs - 1);
  return true;
}

bool assign_inplace_numeric_enabled() {
  // Single runtime policy: keep in-place numeric updates enabled.
  return true;
}

bool assign_direct_numeric_binary_enabled() {
  static const bool enabled = env_flag_enabled("SPARK_ASSIGN_DIRECT_NUMERIC_BINARY", true);
  return enabled;
}

bool recurrence_quickening_enabled() {
  static const bool enabled = env_flag_enabled("SPARK_ASSIGN_RECURRENCE_CACHE", true);
  return enabled;
}

bool recurrence_arith_quickening_enabled() {
  static const bool enabled = env_flag_enabled("SPARK_ASSIGN_RECURRENCE_ARITH", true);
  return enabled;
}

bool should_use_recurrence_quickening(const Value& current, BinaryOp op) {
  if (!is_numeric_kind(current)) {
    return false;
  }
  if (op == BinaryOp::Div || op == BinaryOp::Mod || op == BinaryOp::Pow) {
    return true;
  }
  if (op != BinaryOp::Add && op != BinaryOp::Sub && op != BinaryOp::Mul) {
    return false;
  }
  if (!recurrence_arith_quickening_enabled()) {
    return false;
  }

  if (current.kind == Value::Kind::Int || current.kind == Value::Kind::Double) {
    return true;
  }
  if (current.kind == Value::Kind::Numeric && current.numeric_value.has_value()) {
    return true;
  }
  return false;
}

bool is_numeric_arithmetic_op(BinaryOp op) {
  return op == BinaryOp::Add || op == BinaryOp::Sub || op == BinaryOp::Mul ||
         op == BinaryOp::Div || op == BinaryOp::Mod || op == BinaryOp::Pow;
}

bool assign_direct_numeric_kind_supported(const Value& current) {
  if (current.kind == Value::Kind::Int || current.kind == Value::Kind::Double) {
    return true;
  }
  if (current.kind != Value::Kind::Numeric || !current.numeric_value) {
    return false;
  }
  // Keep direct in-place binary path enabled for all numeric primitives.
  // High-precision lanes still preserve correctness via MPFR helpers.
  return true;
}

long long assign_fast_bench_tick_i64() {
#if defined(__APPLE__) && defined(__aarch64__)
  std::uint64_t ticks = 0;
  static const std::uint64_t freq = []() {
    std::uint64_t out = 0;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(out));
    return out == 0U ? 1U : out;
  }();
  asm volatile("mrs %0, cntvct_el0" : "=r"(ticks));
  const auto ns =
      static_cast<std::uint64_t>((static_cast<unsigned __int128>(ticks) * 1000000000ULL) / freq);
  return static_cast<long long>(ns);
#elif defined(__APPLE__)
  static const mach_timebase_info_data_t timebase = [] {
    mach_timebase_info_data_t info{};
    mach_timebase_info(&info);
    if (info.denom == 0U) {
      info.numer = 1U;
      info.denom = 1U;
    }
    return info;
  }();
  static const bool one_to_one = (timebase.numer == 1U && timebase.denom == 1U);
  static const long double tick_to_ns =
      static_cast<long double>(timebase.numer) / static_cast<long double>(timebase.denom);
  const std::uint64_t ticks = mach_absolute_time();
  if (one_to_one) {
    return static_cast<long long>(ticks);
  }
  return static_cast<long long>(static_cast<long double>(ticks) * tick_to_ns);
#elif defined(CLOCK_MONOTONIC_RAW)
  struct timespec ts {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return static_cast<long long>(ts.tv_sec) * 1000000000LL + static_cast<long long>(ts.tv_nsec);
#else
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<long long>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
#endif
}

long long assign_fast_bench_tick_raw_i64() {
#if defined(__APPLE__) && defined(__aarch64__)
  std::uint64_t ticks = 0;
  asm volatile("mrs %0, cntvct_el0" : "=r"(ticks));
  return static_cast<long long>(ticks);
#elif defined(__APPLE__)
  return static_cast<long long>(mach_absolute_time());
#elif defined(CLOCK_MONOTONIC_RAW)
  struct timespec ts {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return static_cast<long long>(ts.tv_sec) * 1000000000LL + static_cast<long long>(ts.tv_nsec);
#else
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<long long>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
#endif
}

long long assign_fast_bench_tick_scale_num_i64() {
#if defined(__APPLE__) && defined(__aarch64__)
  return 1000000000LL;
#elif defined(__APPLE__)
  static const mach_timebase_info_data_t timebase = [] {
    mach_timebase_info_data_t info{};
    mach_timebase_info(&info);
    if (info.denom == 0U) {
      info.numer = 1U;
      info.denom = 1U;
    }
    return info;
  }();
  return static_cast<long long>(timebase.numer);
#else
  return 1LL;
#endif
}

long long assign_fast_bench_tick_scale_den_i64() {
#if defined(__APPLE__) && defined(__aarch64__)
  static const std::uint64_t freq = []() {
    std::uint64_t out = 0;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(out));
    return out == 0U ? 1U : out;
  }();
  return static_cast<long long>(freq);
#elif defined(__APPLE__)
  static const mach_timebase_info_data_t timebase = [] {
    mach_timebase_info_data_t info{};
    mach_timebase_info(&info);
    if (info.denom == 0U) {
      info.numer = 1U;
      info.denom = 1U;
    }
    return info;
  }();
  return static_cast<long long>(timebase.denom);
#else
  return 1LL;
#endif
}

bool assign_try_eval_fast_noarg_builtin(const Value::Builtin& builtin, Value& out) {
  if (builtin.name == "bench_tick") {
    out = Value::int_value_of(assign_fast_bench_tick_i64());
    return true;
  }
  if (builtin.name == "bench_tick_raw") {
    out = Value::int_value_of(assign_fast_bench_tick_raw_i64());
    return true;
  }
  if (builtin.name == "bench_tick_scale_num") {
    out = Value::int_value_of(assign_fast_bench_tick_scale_num_i64());
    return true;
  }
  if (builtin.name == "bench_tick_scale_den") {
    out = Value::int_value_of(assign_fast_bench_tick_scale_den_i64());
    return true;
  }
  return false;
}

bool assign_direct_operands_compatible(const Value& current, const Value& left,
                                       const Value& right, BinaryOp op) {
  if (current.kind == Value::Kind::Int) {
    return left.kind == Value::Kind::Int && right.kind == Value::Kind::Int;
  }
  if (current.kind == Value::Kind::Double) {
    const auto scalar_like = [](const Value& v) {
      return v.kind == Value::Kind::Int || v.kind == Value::Kind::Double;
    };
    return scalar_like(left) && scalar_like(right);
  }
  if (current.kind == Value::Kind::Numeric && current.numeric_value) {
    if (!is_numeric_kind(left) || !is_numeric_kind(right)) {
      return false;
    }
    const auto current_kind = current.numeric_value->kind;
    const auto left_kind = runtime_numeric_kind(left);
    const auto right_kind = runtime_numeric_kind(right);
    if (left_kind == current_kind && right_kind == current_kind) {
      return true;
    }
    // Allow direct in-place path when operation result kind already matches slot kind.
    const auto result_kind = promote_result_kind(op, left_kind, right_kind);
    return result_kind == current_kind;
  }
  return false;
}

const VariableExpr* as_variable_expr_assign(const Expr* expr) {
  if (!expr || expr->kind != Expr::Kind::Variable) {
    return nullptr;
  }
  return static_cast<const VariableExpr*>(expr);
}

const CallExpr* as_call_expr_assign(const Expr* expr) {
  if (!expr || expr->kind != Expr::Kind::Call) {
    return nullptr;
  }
  return static_cast<const CallExpr*>(expr);
}

const NumberExpr* as_number_expr_assign(const Expr* expr) {
  if (!expr || expr->kind != Expr::Kind::Number) {
    return nullptr;
  }
  return static_cast<const NumberExpr*>(expr);
}

const UnaryExpr* as_unary_expr_assign(const Expr* expr) {
  if (!expr || expr->kind != Expr::Kind::Unary) {
    return nullptr;
  }
  return static_cast<const UnaryExpr*>(expr);
}

const BinaryExpr* as_binary_expr_assign(const Expr* expr) {
  if (!expr || expr->kind != Expr::Kind::Binary) {
    return nullptr;
  }
  return static_cast<const BinaryExpr*>(expr);
}

bool parse_int_numeric_constructor_call_assign(const Expr* expr, const Expr*& out_arg) {
  const auto* call = as_call_expr_assign(expr);
  if (!call || !call->callee || call->callee->kind != Expr::Kind::Variable ||
      call->args.size() != 1 || !call->args[0]) {
    return false;
  }
  const auto& callee = static_cast<const VariableExpr&>(*call->callee);
  try {
    const auto kind = numeric_kind_from_name(callee.name);
    if (!numeric_kind_is_int(kind)) {
      return false;
    }
  } catch (const EvalException&) {
    return false;
  }
  out_arg = call->args[0].get();
  return true;
}

bool parse_numeric_constructor_call_assign(const Expr* expr, Value::NumericKind& out_kind,
                                           const Expr*& out_arg) {
  const auto* call = as_call_expr_assign(expr);
  if (!call || !call->callee || call->callee->kind != Expr::Kind::Variable ||
      call->args.size() != 1 || !call->args[0]) {
    return false;
  }
  const auto& callee = static_cast<const VariableExpr&>(*call->callee);
  try {
    out_kind = numeric_kind_from_name(callee.name);
  } catch (const EvalException&) {
    return false;
  }
  out_arg = call->args[0].get();
  return true;
}

bool extract_numeric_literal_text_assign(const Expr& expr, std::string& out_text,
                                         bool& out_is_int) {
  if (expr.kind == Expr::Kind::Number) {
    const auto& number = static_cast<const NumberExpr&>(expr);
    if (!number.raw_text.empty()) {
      out_text = number.raw_text;
      out_is_int = number.is_int;
      return true;
    }
    return false;
  }
  if (expr.kind == Expr::Kind::Unary) {
    const auto& unary = static_cast<const UnaryExpr&>(expr);
    if (unary.op == UnaryOp::Neg && unary.operand &&
        unary.operand->kind == Expr::Kind::Number) {
      const auto& number = static_cast<const NumberExpr&>(*unary.operand);
      if (!number.raw_text.empty()) {
        out_text = "-" + number.raw_text;
        out_is_int = number.is_int;
        return true;
      }
    }
  }
  return false;
}

bool try_numeric_like_to_i128(const Value& value, I128& out) {
  if (value.kind == Value::Kind::Int) {
    out = static_cast<I128>(value.int_value);
    return true;
  }
  if (value.kind == Value::Kind::Numeric && value.numeric_value &&
      numeric_kind_is_int(value.numeric_value->kind) &&
      value.numeric_value->parsed_int_valid) {
    out = value.numeric_value->parsed_int;
    return true;
  }
  return false;
}

bool try_eval_fast_int_expr(const Expr* expr, const std::shared_ptr<Environment>& env, I128& out) {
  if (!expr) {
    return false;
  }
  const Expr* ctor_arg = nullptr;
  if (parse_int_numeric_constructor_call_assign(expr, ctor_arg)) {
    return try_eval_fast_int_expr(ctor_arg, env, out);
  }
  if (const auto* number = as_number_expr_assign(expr)) {
    if (!number->is_int || !std::isfinite(number->value)) {
      return false;
    }
    const long double rounded = std::nearbyint(number->value);
    if (std::fabs(static_cast<long double>(number->value) - rounded) > 1e-12L) {
      return false;
    }
    if (rounded < static_cast<long double>(assign_i128_min()) ||
        rounded > static_cast<long double>(assign_i128_max())) {
      return false;
    }
    out = static_cast<I128>(rounded);
    return true;
  }
  if (const auto* variable = as_variable_expr_assign(expr)) {
    struct FastIntVarCacheEntry {
      const VariableExpr* expr = nullptr;
      std::uint64_t env_id = 0;
      std::size_t values_size = 0;
      std::size_t bucket_count = 0;
      const Value* value = nullptr;
    };
    constexpr std::size_t kFastIntVarCacheSize = 1024;
    static thread_local std::array<FastIntVarCacheEntry, kFastIntVarCacheSize> cache{};

    const auto key_a = static_cast<std::size_t>(
        reinterpret_cast<std::uintptr_t>(variable));
    const auto key_b =
        static_cast<std::size_t>(env ? env->stable_id : 0) * 11400714819323198485ull;
    auto& slot = cache[(key_a ^ key_b) & (kFastIntVarCacheSize - 1)];
    const auto env_id = env ? env->stable_id : 0;
    const auto values_size = env ? env->values.size() : 0U;
    const auto bucket_count = env ? env->values.bucket_count() : 0U;

    const Value* value = nullptr;
    if (slot.expr == variable && slot.env_id == env_id &&
        slot.values_size == values_size && slot.bucket_count == bucket_count &&
        slot.value != nullptr) {
      value = slot.value;
    } else {
      value = env ? env->get_ptr(variable->name) : nullptr;
      slot.expr = variable;
      slot.env_id = env_id;
      slot.values_size = values_size;
      slot.bucket_count = bucket_count;
      slot.value = value;
    }
    if (!value) {
      return false;
    }
    return try_numeric_like_to_i128(*value, out);
  }
  if (const auto* unary = as_unary_expr_assign(expr)) {
    if (unary->op != UnaryOp::Neg || !unary->operand) {
      return false;
    }
    I128 operand = 0;
    if (!try_eval_fast_int_expr(unary->operand.get(), env, operand)) {
      return false;
    }
    if (operand == assign_i128_min()) {
      return false;
    }
    out = -operand;
    return true;
  }
  const auto* binary = as_binary_expr_assign(expr);
  if (!binary) {
    return false;
  }
  I128 lhs = 0;
  I128 rhs = 0;
  if (!try_eval_fast_int_expr(binary->left.get(), env, lhs) ||
      !try_eval_fast_int_expr(binary->right.get(), env, rhs)) {
    return false;
  }

  switch (binary->op) {
    case BinaryOp::Add: {
      if (__builtin_add_overflow(lhs, rhs, &out)) {
        return false;
      }
      return true;
    }
    case BinaryOp::Sub: {
      if (__builtin_sub_overflow(lhs, rhs, &out)) {
        return false;
      }
      return true;
    }
    case BinaryOp::Mul: {
      if (__builtin_mul_overflow(lhs, rhs, &out)) {
        return false;
      }
      return true;
    }
    case BinaryOp::Mod:
      if (rhs == 0) {
        throw EvalException("modulo by zero");
      }
      if (assign_try_fast_mod_pow2(lhs, rhs, out)) {
        return true;
      }
      out = lhs % rhs;
      return true;
    case BinaryOp::Div:
      // Keep canonical semantics for int division (float promotion rules).
      return false;
    case BinaryOp::Pow:
      // Keep canonical semantics for int pow (float promotion rules).
      return false;
    default:
      return false;
  }
}

bool try_assign_fast_int_like_expr(const AssignStmt& assign,
                                   const std::shared_ptr<Environment>& env,
                                   Value& current) {
  if (!assign.value) {
    return false;
  }
  if (current.kind != Value::Kind::Int &&
      (current.kind != Value::Kind::Numeric || !current.numeric_value ||
       !numeric_kind_is_int(current.numeric_value->kind))) {
    return false;
  }
  I128 out = 0;
  if (!try_eval_fast_int_expr(assign.value.get(), env, out)) {
    return false;
  }

  if (current.kind == Value::Kind::Int) {
    if (out < static_cast<I128>(std::numeric_limits<long long>::min()) ||
        out > static_cast<I128>(std::numeric_limits<long long>::max())) {
      return false;
    }
    current.int_value = static_cast<long long>(out);
    return true;
  }

  const auto target_kind = current.numeric_value->kind;
  const auto source = Value::numeric_int_value_of(Value::NumericKind::I128, out);
  if (cast_numeric_to_kind_inplace(target_kind, source, current)) {
    return true;
  }
  const auto converted = cast_numeric_to_kind(target_kind, source);
  if (!copy_numeric_value_inplace(current, converted)) {
    current = converted;
  }
  return true;
}

bool try_inplace_numeric_constructor_assign(const AssignStmt& assign, Interpreter& self,
                                            const std::shared_ptr<Environment>& env,
                                            Value& current) {
  if (!assign.value) {
    return false;
  }
  const auto* call = as_call_expr_assign(assign.value.get());
  if (!call || call->args.size() != 1 || !call->callee ||
      call->callee->kind != Expr::Kind::Variable) {
    return false;
  }
  if (current.kind != Value::Kind::Numeric || !current.numeric_value) {
    return false;
  }

  const auto& callee_var = static_cast<const VariableExpr&>(*call->callee);
  const auto* callee = env->get_ptr(callee_var.name);
  if (!callee || callee->kind != Value::Kind::Builtin || !callee->builtin_value) {
    return false;
  }

  Value::NumericKind target_kind = Value::NumericKind::F64;
  try {
    target_kind = numeric_kind_from_name(callee->builtin_value->name);
  } catch (const EvalException&) {
    return false;
  }

  if (current.numeric_value->kind != target_kind) {
    return false;
  }

  if (numeric_kind_is_int(target_kind)) {
    I128 fast_out = 0;
    if (try_eval_fast_int_expr(call->args[0].get(), env, fast_out)) {
      assign_numeric_int_value_inplace(current, target_kind, fast_out);
      return true;
    }
  }

  Value source;
  if (const auto* arg_var = as_variable_expr_assign(call->args[0].get())) {
    if (const auto* value = env->get_ptr(arg_var->name)) {
      source = *value;
    } else {
      throw EvalException("undefined variable: " + arg_var->name);
    }
  } else {
    source = self.evaluate(*call->args[0], env);
  }
  if (!is_numeric_kind(source)) {
    throw EvalException(callee->builtin_value->name +
                        "() expects exactly one numeric argument");
  }

  if (cast_numeric_to_kind_inplace(target_kind, source, current)) {
    return true;
  }
  const auto converted = cast_numeric_to_kind(target_kind, source);
  if (!copy_numeric_value_inplace(current, converted)) {
    current = converted;
  }
  return true;
}

bool invariant_numeric_operand_stamp(const Value& value, std::uint64_t& out_stamp) {
  switch (value.kind) {
    case Value::Kind::Int: {
      out_stamp = static_cast<std::uint64_t>(value.int_value);
      return true;
    }
    case Value::Kind::Double: {
      std::uint64_t bits = 0;
      std::memcpy(&bits, &value.double_value, sizeof(bits));
      out_stamp = bits;
      return true;
    }
    case Value::Kind::Numeric:
      break;
    default:
      return false;
  }
  if (!value.numeric_value) {
    return false;
  }
  const auto& numeric = *value.numeric_value;
  std::uint64_t stamp = static_cast<std::uint64_t>(numeric.kind) * 0x9e3779b97f4a7c15ULL;
  if (numeric.high_precision_cache) {
    const auto cache_addr =
        reinterpret_cast<std::uintptr_t>(numeric.high_precision_cache.get());
    stamp ^= static_cast<std::uint64_t>(cache_addr >> 4U);
    stamp ^= numeric.revision * 0xbf58476d1ce4e5b9ULL;
    out_stamp = stamp;
    return true;
  }
  if (numeric.parsed_int_valid) {
    const auto v = static_cast<unsigned __int128>(numeric.parsed_int);
    const auto lo = static_cast<std::uint64_t>(v);
    const auto hi = static_cast<std::uint64_t>(v >> 64U);
    stamp ^= lo;
    stamp ^= (hi * 0x94d049bb133111ebULL);
    out_stamp = stamp;
    return true;
  }
  if (numeric.parsed_float_valid) {
    static_assert(sizeof(long double) <= 16, "unexpected long double size");
    std::uint64_t lo = 0;
    std::uint64_t hi = 0;
    std::memcpy(&lo, &numeric.parsed_float, sizeof(std::uint64_t));
    if (sizeof(long double) > sizeof(std::uint64_t)) {
      std::memcpy(&hi, reinterpret_cast<const std::uint8_t*>(&numeric.parsed_float) + sizeof(std::uint64_t),
                  sizeof(long double) - sizeof(std::uint64_t));
    }
    stamp ^= lo;
    stamp ^= (hi * 0x94d049bb133111ebULL);
    out_stamp = stamp;
    return true;
  }
  out_stamp = stamp ^ numeric.revision;
  return true;
}

bool high_precision_operand_stamp_fast(const Value& value, std::uint64_t& out_stamp) {
  if (value.kind != Value::Kind::Numeric || !value.numeric_value) {
    return false;
  }
  const auto& numeric = *value.numeric_value;
  if (!numeric_kind_is_high_precision_float(numeric.kind)) {
    return false;
  }
  const auto cache_addr =
      reinterpret_cast<std::uintptr_t>(numeric.high_precision_cache.get());
  out_stamp = (static_cast<std::uint64_t>(numeric.kind) * 0x9e3779b97f4a7c15ULL) ^
              static_cast<std::uint64_t>(cache_addr >> 4U) ^
              (numeric.revision * 0xbf58476d1ce4e5b9ULL);
  return true;
}

bool numeric_values_equal_fast(const Value& lhs, const Value& rhs) {
  if (lhs.kind != rhs.kind) {
    return false;
  }
  if (lhs.kind == Value::Kind::Int) {
    return lhs.int_value == rhs.int_value;
  }
  if (lhs.kind == Value::Kind::Double) {
    return lhs.double_value == rhs.double_value;
  }
  if (lhs.kind != Value::Kind::Numeric || !lhs.numeric_value || !rhs.numeric_value) {
    return false;
  }

  const auto& ln = *lhs.numeric_value;
  const auto& rn = *rhs.numeric_value;
  if (ln.kind != rn.kind) {
    return false;
  }

#if defined(SPARK_HAS_MPFR)
  if (numeric_kind_is_high_precision_float(ln.kind)) {
    if (ln.high_precision_cache && rn.high_precision_cache &&
        ln.high_precision_cache.get() == rn.high_precision_cache.get()) {
      return true;
    }
  }
#endif

  if (ln.parsed_int_valid && rn.parsed_int_valid) {
    return ln.parsed_int == rn.parsed_int;
  }
  if (ln.parsed_float_valid && rn.parsed_float_valid) {
    return ln.parsed_float == rn.parsed_float;
  }
  if (!ln.payload.empty() && !rn.payload.empty()) {
    return ln.payload == rn.payload;
  }
  return false;
}

const Value* try_resolve_assign_operand(const Expr& expr, Interpreter& self,
                                        const std::shared_ptr<Environment>& env,
                                        Value& temp) {
  if (expr.kind == Expr::Kind::Variable) {
    const auto& variable = static_cast<const VariableExpr&>(expr);
    struct VarRefCacheEntry {
      const Expr* expr = nullptr;
      std::uint64_t env_id = 0;
      std::uint64_t values_epoch = 0;
      const Value* value = nullptr;
    };
    constexpr std::size_t kVarRefCacheSize = 512;
    static thread_local std::array<VarRefCacheEntry, kVarRefCacheSize> cache{};
    const auto raw_hash = static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(&expr)) ^
                          (static_cast<std::size_t>(env->stable_id) * 11400714819323198485ull);
    auto& slot = cache[raw_hash & (kVarRefCacheSize - 1)];
    if (slot.expr == &expr && slot.env_id == env->stable_id &&
        slot.values_epoch == env->values_epoch && slot.value != nullptr) {
      return slot.value;
    }
    if (const auto* ref = env->get_ptr(variable.name); ref != nullptr) {
      slot.expr = &expr;
      slot.env_id = env->stable_id;
      slot.values_epoch = env->values_epoch;
      slot.value = ref;
      return ref;
    }
  }
  if (expr.kind == Expr::Kind::Number) {
    const auto& number = static_cast<const NumberExpr&>(expr);
    temp = number.is_int ? Value::int_value_of(static_cast<long long>(number.value))
                         : Value::double_value_of(number.value);
    return &temp;
  }
  {
    Value::NumericKind target_kind = Value::NumericKind::F64;
    const Expr* ctor_arg = nullptr;
    if (parse_numeric_constructor_call_assign(&expr, target_kind, ctor_arg) && ctor_arg) {
      std::string literal_text;
      bool literal_is_int = false;
      if (extract_numeric_literal_text_assign(*ctor_arg, literal_text, literal_is_int)) {
        const auto source_kind =
            literal_is_int ? Value::NumericKind::I512 : Value::NumericKind::F512;
        temp = cast_numeric_to_kind(target_kind,
                                    Value::numeric_value_of(source_kind, literal_text));
        return &temp;
      }
      if (const auto* arg_var = as_variable_expr_assign(ctor_arg)) {
        if (const auto* source = env ? env->get_ptr(arg_var->name) : nullptr;
            source && is_numeric_kind(*source)) {
          temp = cast_numeric_to_kind(target_kind, *source);
          return &temp;
        }
      }
    }
  }
  temp = self.evaluate(expr, env);
  return &temp;
}

struct InvariantNumericAssignCacheEntry {
  const Expr* expr = nullptr;
  std::uint64_t env_id = 0;
  Value* target_ptr = nullptr;
  const Value* left_ptr = nullptr;
  const Value* right_ptr = nullptr;
  bool left_revision_valid = false;
  bool right_revision_valid = false;
  std::uint64_t left_revision = 0;
  std::uint64_t right_revision = 0;
  std::uint64_t left_stamp = 0;
  std::uint64_t right_stamp = 0;
  std::uint64_t result_stamp = 0;
  Value result = Value::nil();
};

struct RecurrenceNumericAssignCacheEntry {
  const Expr* expr = nullptr;
  std::uint64_t env_id = 0;
  Value* target_ptr = nullptr;
  BinaryOp op = BinaryOp::Add;
  bool target_on_left = true;
  bool rhs_is_variable = false;
  std::string rhs_name;
  Value rhs_literal = Value::nil();
  bool rhs_revision_valid = false;
  std::uint64_t rhs_revision = 0;
  std::uint64_t rhs_stamp = 0;
};

InvariantNumericAssignCacheEntry& invariant_numeric_assign_cache_slot(const Expr* expr,
                                                                      std::uint64_t env_id) {
  constexpr std::size_t kCacheSize = 1024;
  static thread_local std::array<InvariantNumericAssignCacheEntry, kCacheSize> cache{};
  const auto raw_hash = static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(expr)) ^
                        (static_cast<std::size_t>(env_id) * 11400714819323198485ull);
  return cache[raw_hash & (kCacheSize - 1)];
}

RecurrenceNumericAssignCacheEntry& recurrence_numeric_assign_cache_slot(const Expr* expr,
                                                                        std::uint64_t env_id) {
  constexpr std::size_t kCacheSize = 1024;
  static thread_local std::array<RecurrenceNumericAssignCacheEntry, kCacheSize> cache{};
  const auto raw_hash = static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(expr)) ^
                        (static_cast<std::size_t>(env_id) * 1469598103934665603ull);
  return cache[raw_hash & (kCacheSize - 1)];
}

bool try_copy_cached_numeric_result(const Value& cached, Value& target) {
  return copy_numeric_value_inplace(target, cached);
}

bool try_apply_recurrence_assign_cache_fast(const AssignStmt& assign,
                                            const std::shared_ptr<Environment>& env,
                                            Value* current) {
  if (!assign.value || !current) {
    return false;
  }
  auto& slot = recurrence_numeric_assign_cache_slot(assign.value.get(), env->stable_id);
  if (slot.expr != assign.value.get() || slot.env_id != env->stable_id ||
      slot.target_ptr != current) {
    return false;
  }

  const Value* rhs_ptr = nullptr;
  if (slot.rhs_is_variable) {
    rhs_ptr = env->get_ptr(slot.rhs_name);
  } else {
    rhs_ptr = &slot.rhs_literal;
  }
  if (!rhs_ptr || !is_numeric_kind(*rhs_ptr) || !is_numeric_kind(*current)) {
    return false;
  }

  if (slot.rhs_revision_valid) {
    if (rhs_ptr->kind != Value::Kind::Numeric || !rhs_ptr->numeric_value ||
        rhs_ptr->numeric_value->revision != slot.rhs_revision) {
      return false;
    }
  } else {
    std::uint64_t rhs_stamp = 0;
    if (!high_precision_operand_stamp_fast(*rhs_ptr, rhs_stamp) &&
        !invariant_numeric_operand_stamp(*rhs_ptr, rhs_stamp)) {
      return false;
    }
    if (rhs_stamp != slot.rhs_stamp) {
      return false;
    }
  }

  const Value* left = slot.target_on_left ? current : rhs_ptr;
  const Value* right = slot.target_on_left ? rhs_ptr : current;
  if (eval_numeric_binary_value_inplace(slot.op, *left, *right, *current)) {
    if (slot.rhs_revision_valid && rhs_ptr->kind == Value::Kind::Numeric &&
        rhs_ptr->numeric_value) {
      slot.rhs_revision = rhs_ptr->numeric_value->revision;
    }
    return true;
  }
  if (is_numeric_kind(*left) && is_numeric_kind(*right)) {
    *current = eval_numeric_binary_value(slot.op, *left, *right);
    if (slot.rhs_revision_valid && rhs_ptr->kind == Value::Kind::Numeric &&
        rhs_ptr->numeric_value) {
      slot.rhs_revision = rhs_ptr->numeric_value->revision;
    }
    return true;
  }
  return false;
}

void update_recurrence_assign_cache(const AssignStmt& assign,
                                    const std::shared_ptr<Environment>& env,
                                    Value* current, BinaryOp op, bool target_on_left,
                                    const VariableExpr* rhs_var,
                                    const Value& rhs_value) {
  auto& slot = recurrence_numeric_assign_cache_slot(assign.value.get(), env->stable_id);
  slot.expr = assign.value.get();
  slot.env_id = env->stable_id;
  slot.target_ptr = current;
  slot.op = op;
  slot.target_on_left = target_on_left;
  slot.rhs_is_variable = rhs_var != nullptr;
  slot.rhs_name = rhs_var ? rhs_var->name : std::string();
  if (!slot.rhs_is_variable) {
    slot.rhs_literal = rhs_value;
  } else {
    slot.rhs_literal = Value::nil();
  }
  slot.rhs_revision_valid = rhs_value.kind == Value::Kind::Numeric &&
                            rhs_value.numeric_value.has_value();
  slot.rhs_revision = slot.rhs_revision_valid ? rhs_value.numeric_value->revision : 0;
  slot.rhs_stamp = 0;
  if (!slot.rhs_revision_valid) {
    invariant_numeric_operand_stamp(rhs_value, slot.rhs_stamp);
  }
}

bool try_apply_invariant_assign_cache_fast(const AssignStmt& assign,
                                           const std::shared_ptr<Environment>& env,
                                           Value* current) {
  if (!assign.value || !current) {
    return false;
  }
  auto& slot = invariant_numeric_assign_cache_slot(assign.value.get(), env->stable_id);
  if (slot.expr != assign.value.get() || slot.env_id != env->stable_id ||
      slot.target_ptr != current || slot.left_ptr == nullptr || slot.right_ptr == nullptr) {
    return false;
  }

  std::uint64_t left_stamp = 0;
  std::uint64_t right_stamp = 0;
  if (slot.left_revision_valid) {
    if (slot.left_ptr->kind != Value::Kind::Numeric || !slot.left_ptr->numeric_value ||
        slot.left_ptr->numeric_value->revision != slot.left_revision) {
      return false;
    }
    left_stamp = slot.left_stamp;
  } else {
    if (!high_precision_operand_stamp_fast(*slot.left_ptr, left_stamp) &&
        !invariant_numeric_operand_stamp(*slot.left_ptr, left_stamp)) {
      return false;
    }
  }
  if (slot.right_revision_valid) {
    if (slot.right_ptr->kind != Value::Kind::Numeric || !slot.right_ptr->numeric_value ||
        slot.right_ptr->numeric_value->revision != slot.right_revision) {
      return false;
    }
    right_stamp = slot.right_stamp;
  } else {
    if (!high_precision_operand_stamp_fast(*slot.right_ptr, right_stamp) &&
        !invariant_numeric_operand_stamp(*slot.right_ptr, right_stamp)) {
      return false;
    }
  }
  if (left_stamp != slot.left_stamp || right_stamp != slot.right_stamp) {
    return false;
  }

  if (numeric_values_equal_fast(*current, slot.result)) {
    return true;
  }

  std::uint64_t current_stamp = 0;
  if ((high_precision_operand_stamp_fast(*current, current_stamp) ||
       invariant_numeric_operand_stamp(*current, current_stamp)) &&
      current_stamp == slot.result_stamp) {
    return true;
  }

  if (try_copy_cached_numeric_result(slot.result, *current)) {
    return true;
  }
  *current = slot.result;
  return true;
}

}  // namespace

Value execute_case_assign(const AssignStmt& assign, Interpreter& self,
                         const std::shared_ptr<Environment>& env) {
  if (assign.target->kind == Expr::Kind::Variable) {
    const auto& variable = static_cast<const VariableExpr&>(*assign.target);
    const bool inplace_numeric_assign = assign_inplace_numeric_enabled();
    auto* current = [&]() -> Value* {
      struct AssignTargetCacheEntry {
        const AssignStmt* stmt = nullptr;
        std::uint64_t env_id = 0;
        std::uint64_t values_epoch = 0;
        Value* ptr = nullptr;
      };
      constexpr std::size_t kAssignTargetCacheSize = 1024;
      static thread_local std::array<AssignTargetCacheEntry, kAssignTargetCacheSize> cache{};
      const auto key_a =
          static_cast<std::size_t>(reinterpret_cast<std::uintptr_t>(&assign));
      const auto key_b =
          static_cast<std::size_t>(env->stable_id) * 11400714819323198485ull;
      auto& slot = cache[(key_a ^ key_b) & (kAssignTargetCacheSize - 1)];
      if (slot.stmt == &assign && slot.env_id == env->stable_id &&
          slot.values_epoch == env->values_epoch && slot.ptr != nullptr) {
        return slot.ptr;
      }
      auto* ptr = env->get_ptr(variable.name);
      slot.stmt = &assign;
      slot.env_id = env->stable_id;
      slot.values_epoch = env->values_epoch;
      slot.ptr = ptr;
      return ptr;
    }();

    if (current != nullptr) {
      // Generic fast path: assignment from no-arg builtin call avoids full
      // expression dispatch and argument vector construction.
      if (assign.value) {
        if (const auto* call = as_call_expr_assign(assign.value.get());
            call && call->args.empty() && call->callee &&
            call->callee->kind == Expr::Kind::Variable) {
          const auto& callee_var = static_cast<const VariableExpr&>(*call->callee);
          if (const auto* callee = env->get_ptr(callee_var.name);
              callee && callee->kind == Value::Kind::Builtin && callee->builtin_value) {
            Value out = Value::nil();
            if (!assign_try_eval_fast_noarg_builtin(*callee->builtin_value, out)) {
              static const std::vector<Value> kNoArgs;
              out = callee->builtin_value->impl(kNoArgs);
            }
            if (!copy_numeric_value_inplace(*current, out)) {
              *current = std::move(out);
            }
            return Value::nil();
          }
        }

        // Generic in-place numeric constructor path:
        // x = fN(expr) where x already has kind fN.
        if (try_inplace_numeric_constructor_assign(assign, self, env, *current)) {
          return Value::nil();
        }

        // Fast integer-expression assignment path:
        // evaluate nested int arithmetic trees without full generic dispatch.
        if (try_assign_fast_int_like_expr(assign, env, *current)) {
          return Value::nil();
        }
      }

      if (inplace_numeric_assign && assign.value && assign.value->kind == Expr::Kind::Binary) {
        const auto& binary = static_cast<const BinaryExpr&>(*assign.value);
        if (assign_direct_numeric_binary_enabled() &&
            is_numeric_arithmetic_op(binary.op) &&
            assign_direct_numeric_kind_supported(*current)) {
          const auto* left_var = as_variable_expr_assign(binary.left.get());
          const auto* right_var = as_variable_expr_assign(binary.right.get());
          if (left_var && right_var) {
            const Value* left = env->get_ptr(left_var->name);
            const Value* right = env->get_ptr(right_var->name);
            if (left && right && is_numeric_kind(*left) && is_numeric_kind(*right) &&
                assign_direct_operands_compatible(*current, *left, *right, binary.op)) {
              if (current->kind == Value::Kind::Int &&
                  left->kind == Value::Kind::Int &&
                  right->kind == Value::Kind::Int) {
                switch (binary.op) {
                  case BinaryOp::Add:
                    current->int_value = left->int_value + right->int_value;
                    return Value::nil();
                  case BinaryOp::Sub:
                    current->int_value = left->int_value - right->int_value;
                    return Value::nil();
                  case BinaryOp::Mul:
                    current->int_value = left->int_value * right->int_value;
                    return Value::nil();
                  case BinaryOp::Div:
                    if (right->int_value == 0) {
                      throw EvalException("division by zero");
                    }
                    *current = Value::double_value_of(
                        static_cast<double>(left->int_value) /
                        static_cast<double>(right->int_value));
                    return Value::nil();
                  case BinaryOp::Mod:
                    if (right->int_value == 0) {
                      throw EvalException("modulo by zero");
                    }
                    {
                      I128 fast_mod = 0;
                      if (assign_try_fast_mod_pow2(static_cast<I128>(left->int_value),
                                                   static_cast<I128>(right->int_value),
                                                   fast_mod)) {
                        current->int_value = static_cast<long long>(fast_mod);
                        return Value::nil();
                      }
                    }
                    current->int_value = left->int_value % right->int_value;
                    return Value::nil();
                  case BinaryOp::Pow:
                    break;
                  default:
                    break;
                }
              }
              if (current->kind == Value::Kind::Double &&
                  (left->kind == Value::Kind::Int || left->kind == Value::Kind::Double) &&
                  (right->kind == Value::Kind::Int || right->kind == Value::Kind::Double)) {
                const double lhs = (left->kind == Value::Kind::Int)
                                       ? static_cast<double>(left->int_value)
                                       : left->double_value;
                const double rhs = (right->kind == Value::Kind::Int)
                                       ? static_cast<double>(right->int_value)
                                       : right->double_value;
                switch (binary.op) {
                  case BinaryOp::Add:
                    current->double_value = lhs + rhs;
                    return Value::nil();
                  case BinaryOp::Sub:
                    current->double_value = lhs - rhs;
                    return Value::nil();
                  case BinaryOp::Mul:
                    current->double_value = lhs * rhs;
                    return Value::nil();
                  case BinaryOp::Div:
                    if (rhs == 0.0) {
                      throw EvalException("division by zero");
                    }
                    current->double_value = lhs / rhs;
                    return Value::nil();
                  case BinaryOp::Mod:
                    if (rhs == 0.0) {
                      throw EvalException("modulo by zero");
                    }
                    current->double_value = std::fmod(lhs, rhs);
                    return Value::nil();
                  case BinaryOp::Pow:
                    current->double_value = std::pow(lhs, rhs);
                    return Value::nil();
                  default:
                    break;
                }
              }
              if (eval_numeric_binary_value_inplace(binary.op, *left, *right, *current)) {
                return Value::nil();
              }
              *current = eval_numeric_binary_value(binary.op, *left, *right);
              return Value::nil();
            }
          }
        }

        const bool recurrence_allowed =
            should_use_recurrence_quickening(*current, binary.op) &&
            recurrence_quickening_enabled();

        // Recurrence quickening for generic loops:
        // x = x op y  /  x = y op x
        if (recurrence_allowed &&
            try_apply_recurrence_assign_cache_fast(assign, env, current)) {
          return Value::nil();
        }

        // Hot-path: if this exact assignment already has a validated invariant
        // cache entry (same env/target and same operand stamps), skip full
        // expression resolution and reuse cached result immediately.
        if (try_apply_invariant_assign_cache_fast(assign, env, current)) {
          return Value::nil();
        }

        if (is_numeric_arithmetic_op(binary.op)) {
          Value left_temp = Value::nil();
          Value right_temp = Value::nil();
          const Value* left = try_resolve_assign_operand(*binary.left, self, env, left_temp);
          const Value* right = try_resolve_assign_operand(*binary.right, self, env, right_temp);
          const auto* left_var = as_variable_expr_assign(binary.left.get());
          const auto* right_var = as_variable_expr_assign(binary.right.get());
          const bool recurrence_target_left =
              left_var && left_var->name == variable.name &&
              ((right_var != nullptr) || binary.right->kind == Expr::Kind::Number);
          const bool recurrence_target_right =
              right_var && right_var->name == variable.name &&
              ((left_var != nullptr) || binary.left->kind == Expr::Kind::Number);
          const Value* recurrence_rhs = recurrence_target_left ? right : left;
          const auto* recurrence_rhs_var =
              recurrence_target_left ? right_var : left_var;
          const bool recurrence_numeric =
              (recurrence_target_left || recurrence_target_right) &&
              recurrence_rhs && is_numeric_kind(*current) && is_numeric_kind(*recurrence_rhs);

          const bool invariant_numeric_operands =
              left && right && left != current && right != current &&
              is_numeric_kind(*left) && is_numeric_kind(*right) &&
              ((left_var != nullptr) || binary.left->kind == Expr::Kind::Number) &&
              ((right_var != nullptr) || binary.right->kind == Expr::Kind::Number);
          std::uint64_t left_stamp = 0;
          std::uint64_t right_stamp = 0;
          if (invariant_numeric_operands) {
            if (!invariant_numeric_operand_stamp(*left, left_stamp) ||
                !invariant_numeric_operand_stamp(*right, right_stamp)) {
              left_stamp = 0;
              right_stamp = 0;
            }
            auto& slot = invariant_numeric_assign_cache_slot(assign.value.get(), env->stable_id);
            if (slot.expr == assign.value.get() &&
                slot.env_id == env->stable_id &&
                slot.target_ptr == current &&
                slot.left_ptr == left &&
                slot.right_ptr == right &&
                slot.left_stamp == left_stamp &&
                slot.right_stamp == right_stamp) {
              std::uint64_t current_stamp = 0;
              if (invariant_numeric_operand_stamp(*current, current_stamp) &&
                  current_stamp == slot.result_stamp) {
                return Value::nil();
              }
              *current = slot.result;
              return Value::nil();
            }
          }

          if (left && right &&
              eval_numeric_binary_value_inplace(binary.op, *left, *right, *current)) {
            if (recurrence_numeric) {
              if (recurrence_allowed) {
                update_recurrence_assign_cache(assign, env, current, binary.op,
                                               recurrence_target_left, recurrence_rhs_var,
                                               *recurrence_rhs);
              }
            }
            if (invariant_numeric_operands) {
              auto& slot = invariant_numeric_assign_cache_slot(assign.value.get(), env->stable_id);
              slot.expr = assign.value.get();
              slot.env_id = env->stable_id;
              slot.target_ptr = current;
              slot.left_ptr = left;
              slot.right_ptr = right;
              slot.left_revision_valid =
                  left->kind == Value::Kind::Numeric && left->numeric_value.has_value();
              slot.right_revision_valid =
                  right->kind == Value::Kind::Numeric && right->numeric_value.has_value();
              slot.left_revision =
                  slot.left_revision_valid ? left->numeric_value->revision : 0;
              slot.right_revision =
                  slot.right_revision_valid ? right->numeric_value->revision : 0;
              slot.left_stamp = left_stamp;
              slot.right_stamp = right_stamp;
              slot.result = *current;
              invariant_numeric_operand_stamp(slot.result, slot.result_stamp);
            }
            return Value::nil();
          }

          if (left && right &&
              left->kind == Value::Kind::Numeric &&
              right->kind == Value::Kind::Numeric) {
            *current = eval_numeric_binary_value(binary.op, *left, *right);
            if (recurrence_numeric) {
              if (recurrence_allowed) {
                update_recurrence_assign_cache(assign, env, current, binary.op,
                                               recurrence_target_left, recurrence_rhs_var,
                                               *recurrence_rhs);
              }
            }
            if (invariant_numeric_operands) {
              auto& slot = invariant_numeric_assign_cache_slot(assign.value.get(), env->stable_id);
              slot.expr = assign.value.get();
              slot.env_id = env->stable_id;
              slot.target_ptr = current;
              slot.left_ptr = left;
              slot.right_ptr = right;
              slot.left_revision_valid =
                  left->kind == Value::Kind::Numeric && left->numeric_value.has_value();
              slot.right_revision_valid =
                  right->kind == Value::Kind::Numeric && right->numeric_value.has_value();
              slot.left_revision =
                  slot.left_revision_valid ? left->numeric_value->revision : 0;
              slot.right_revision =
                  slot.right_revision_valid ? right->numeric_value->revision : 0;
              slot.left_stamp = left_stamp;
              slot.right_stamp = right_stamp;
              slot.result = *current;
              invariant_numeric_operand_stamp(slot.result, slot.result_stamp);
            }
            return Value::nil();
          }

          if (left && right) {
            *current = self.eval_binary(binary.op, *left, *right);
            return Value::nil();
          }

          return Value::nil();
        }
      }
      auto evaluated = self.evaluate(*assign.value, env);
      if (!copy_numeric_value_inplace(*current, evaluated)) {
        *current = std::move(evaluated);
      }
      return Value::nil();
    }
    auto value = self.evaluate(*assign.value, env);
    env->define(variable.name, value);
    return Value::nil();
  }

  if (assign.target->kind == Expr::Kind::Index) {
    auto value = self.evaluate(*assign.value, env);
    const auto target = flatten_index_target(*assign.target);
    if (!target.variable) {
      throw EvalException("invalid assignment target");
    }
    auto current = env->get(target.variable->name);
    const ExprEvaluator evaluator = [&self](const Expr& expr, const std::shared_ptr<Environment>& local_env) {
      return self.evaluate(expr, local_env);
    };
    assign_indexed_expression(evaluator, current, target.indices, 0, env, value);
    if (!env->set(target.variable->name, current)) {
      throw EvalException("cannot assign to undefined variable: " + target.variable->name);
    }
    return Value::nil();
  }
  throw EvalException("invalid assignment target");
}

}  // namespace spark
