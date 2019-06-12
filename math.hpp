#pragma once
#include <math.h>
#include <cstdint>

namespace ssa
{
	typedef int8_t i8;
	typedef int16_t i16;
	typedef int32_t i32;
	typedef int64_t i64;

	typedef uint8_t u8;
	typedef uint16_t u6;
	typedef uint32_t u32;
	typedef uint64_t u64;

	typedef float f32;
	typedef double f64;

	// modified from: https://stackoverflow.com/questions/13816850/is-it-possible-to-develop-static-for-loop-in-c
	// and: https://stackoverflow.com/questions/53522781/is-there-a-way-to-pass-a-constexpr-value-into-lambda-so-that-it-remains-const
	template <typename size_type, size_type First, size_type Last, typename Functor, size_type Incr = 1>
	constexpr void static_for(Functor&& f)
	{
		if constexpr (First < Last)
		{
			f(std::integral_constant<size_type, First>{});
			static_for<size_type, First + Incr, Last>(std::forward<Functor>(f));
		}
	}


	template<typename T, typename BinaryFunction = T(const T&, const T&), typename...Ts>
	constexpr T foldl(const BinaryFunction& op, const T& identity, const T& head)
	{
		return op(identity, head);
	}

	template<typename T, typename BinaryFunction = T(const T&, const T&), typename...Ts>
	constexpr T foldl(const BinaryFunction& op, const T& identity, const T& head, const Ts&...tail)
	{
		return foldl(op, op(identity, head), tail...);
	}

	template<typename T, typename BinaryFunction = T(const T&, const T&), typename...Ts>
	constexpr T foldr(const BinaryFunction& op, const T& identity, const T& head)
	{
		return op(head, identity);
	}

	template<typename T, typename BinaryFunction = T(const T&, const T&), typename...Ts>
	constexpr T foldr(const BinaryFunction& op, const T& identity, const T& head, const Ts&...tail)
	{
		return op(head, foldr(op, identity, tail...));
	}

	// return +1 if in>0, 0 for in==0, and -1 for in<0
	template<typename T, std::enable_if_t<std::is_pod_v<T>, bool> = true>
	constexpr int sign(const T& in)
	{
		return static_cast<int>(in > 0) - static_cast<int>(in < 0);
	}

	template<typename T>
	constexpr T max(const T& lhs, const T& rhs)
	{
		return lhs >= rhs ? lhs : rhs;
	}

	template<typename T, typename...Ts>
	constexpr T max(const T& head, const Ts&... tail)
	{
		return foldl(max<T>, head, tail...);
	}

	template<typename T>
	constexpr T min(const T& lhs, const T& rhs)
	{
		return lhs <= rhs ? lhs : rhs;
	}

	template<typename T, typename...Ts>
	constexpr T min(const T& head, const Ts&... tail)
	{
		return foldl(min<T>, head, tail...);
	}

	template<typename T>
	constexpr T add(const T& lhs, const T& rhs)
	{
		return lhs + rhs;
	}

	template<typename T, typename ...Ts>
	constexpr T sum(const T& head, const Ts&...tail)
	{
		return foldl(add<T>(), head, tail...);
	}

	template<typename T>
	constexpr T multiply(const T& lhs, const T& rhs)
	{
		return lhs * rhs;
	}

	template<typename T, typename ...Ts>
	constexpr T product(const T& head, const Ts&...tail)
	{
		return foldl(multiply<T>, head, tail...);
	}



	// template friendly inference redefinitions
	// the enable_ifs serve both to resolve ambiguities and for correctness

	template<typename T, std::enable_if_t<std::is_pod_v<T>, bool> = true>
	constexpr T abs(const T& in)
	{
		return ::abs(in);
	}

	template<typename T, std::enable_if_t<std::is_pod_v<T>, bool> = true>
	constexpr T exp(const T& in)
	{
		return ::exp(in);
	}

	template<typename T, std::enable_if_t<std::is_pod_v<T>, bool> = true>
	constexpr T exp2(const T& in)
	{
		return ::exp2(in);
	}

	template<typename T, std::enable_if_t<std::is_pod_v<T>, bool> = true>
	constexpr T log(const T& in)
	{
		return ::log(in);
	}

	template<typename T, std::enable_if_t<std::is_pod_v<T>, bool> = true>
	constexpr T log2(const T& in)
	{
		return ::log2(in);
	}

	template<typename T, std::enable_if_t<std::is_pod_v<T>, bool> = true>
	constexpr T log10(const T& in)
	{
		return ::log10(in);
	}

	template<typename T, typename U, std::enable_if_t<std::is_pod_v<U> && std::is_pod_v<T>, bool> = true>
	constexpr T pow(const T& lhs, const U& rhs)
	{
		return ::pow(lhs, rhs);
	}

	template<typename T, std::enable_if_t<std::is_pod_v<T>, bool> = true>
	constexpr T pow(const T& lhs, const T& rhs)
	{
		return ::pow(lhs, rhs);
	}

	template<typename T, std::enable_if_t<std::is_pod_v<T>, bool> = true>
	constexpr T sqrt(const T& in)
	{
		return ::sqrt(in);
	}

	// cube root
	template<typename T, std::enable_if_t<std::is_pod_v<T>, bool> = true>
	constexpr T cbrt(const T& in)
	{
		return ::cbrt(in);
	}

	template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	constexpr bool isinf(const T& in)
	{
		return ::isinf(in);
	}

	// sqrt(negative) || 0/0
	template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	constexpr bool isnan(const T& in)
	{
		return ::isnan(in);
	}

	// !isnan() && !isinfinite()
	template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	constexpr bool isfinite(const T& in)
	{
		return ::isfinite(in);
	}

	// !(isnan() || isinf() || iszero() || issubnormal())
	template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	constexpr bool isnormal(const T& in)
	{
		return ::isnormal(in);
	}

	// FP_INFINITE, FP_NAN, FP_ZERO, FP_SUBNORMAL, FP_NORMAL
	template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	constexpr int fpclassify(const T& in)
	{
		return ::fpclassify(in);
	}

	// returns true for negative values
	template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	constexpr bool signbit(const T& in)
	{
		return ::signbit(in);
	}

	// rounds to the closest integral value, halfway cases round away from 0
	template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	constexpr T round(const T& in)
	{
		return ::round(in);
	}

	// rounds upwards
	template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	constexpr T ceil(const T& in)
	{
		return ::ceil(in);
	}

	// rounds downwards
	template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	constexpr T floor(const T& in)
	{
		return ::floor(in);
	}

	// rounds towards 0
	template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
	constexpr T trunc(const T& in)
	{
		return ::trunc(in);
	}

}