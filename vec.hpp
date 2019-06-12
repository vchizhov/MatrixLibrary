#pragma once

#pragma once
#include <type_traits>
#include <functional>
#include "math.hpp"
#include "typedef.hpp"

namespace ssa
{
	/*template<typename T, size_t...D>
	struct vec
	{
		static_assert(sizeof...(D) <= 1, "vec:: Supports only one parameter for dimenions.");
	};*/


	/////////////////////////////////////////////////////////////////////////////////////////////
	// Convenience base classes that allows customization of the vectors of dimensions 0,1,2,3,4
	/////////////////////////////////////////////////////////////////////////////////////////////
	template<typename T, size_t D>
	struct BaseVec
	{
		BaseVec() {}
		T e[D];
	};

	// Disallow vectors with 0 coordinates
	template<typename T>
	struct BaseVec<T, 0>
	{
		static_assert(true, "vec::Base:: vector of dimension 0 disallowed.");
	};


	template<typename T>
	struct BaseVec<T, 1>
	{
		BaseVec() {}
		union
		{
			T e[1];
			T x;
			T r;
			T s;
		};
	};

	template<typename T>
	struct BaseVec<T, 2>
	{
		BaseVec() {}

		/*template<typename U, typename V>
		Base(const U& x, const V& y) : x(static_cast<T>(x)), y(static_cast<T>(y)) {}*/

		template<typename U>
		BaseVec(const BaseVec<U, 2>& in) : x(static_cast<T>(in.x)), y(static_cast<T>(in.y)) {}
		union {
			T e[2];
			struct
			{
				T x, y;
			};
			struct
			{
				T r, g;
			};
			struct
			{
				T s, t;
			};
		};
	};

	template<typename T>
	struct BaseVec<T, 3>
	{
		BaseVec() {}

		/*template<typename U, typename V, typename W>
		Base(const U& x, const V& y, const W& z) : x(static_cast<T>(x)), y(static_cast<T>(y)), z(static_cast<T>(z)) {}*/

		template<typename U>
		BaseVec(const BaseVec<U, 3>& in) : x(static_cast<T>(in.x)), y(static_cast<T>(in.y)), z(static_cast<T>(in.z)) {}

		template<typename U, typename V>
		BaseVec(const U& x, const BaseVec<V, 2>& yz) : x(static_cast<T>(x)), yz(yz) {}

		template<typename U, typename V>
		BaseVec(const BaseVec<U, 2>& xy, const V& z) : x(xy), z(static_cast<T>(z)) {}
		union
		{
			T e[3];
			struct
			{
				T x, y, z;
			};
			struct
			{
				BaseVec<T, 2> xy;
				T z;
			};
			struct
			{
				T x;
				BaseVec<T, 2> yz;
			};
			struct
			{
				T r, g, b;
			};
			struct
			{
				BaseVec<T, 2> rg;
				T b;
			};
			struct
			{
				T r;
				BaseVec<T, 2> gb;
			};
			struct
			{
				T s, t, p;
			};
			struct
			{
				BaseVec<T, 2> st;
				T p;
			};
			struct
			{
				T s;
				BaseVec<T, 2> tp;
			};
		};
	};

	template<typename T>
	struct BaseVec<T, 4>
	{
		BaseVec() {}

		/*template<typename U>
		Base(const Base<U, 4>& in) : x(static_cast<T>(in.x)), y(static_cast<T>(in.y)), z(static_cast<T>(in.z)), w(static_cast<T>(in.w)) {}*/

		template<typename U, typename V, typename W>
		BaseVec(const U& x, const BaseVec<V, 2>& yz, const W& w) : x(static_cast<T>(x)), yz(yz), w(static_cast<T>(w)) {}

		template<typename U, typename V>
		BaseVec(const BaseVec<U, 2>& xy, const BaseVec<V, 2>& zw) : xy(xy), zw(zw) {}

		template<typename U, typename V>
		BaseVec(const U& x, const BaseVec<V, 3>& yzw) : x(static_cast<T>(x)), yzw(yzw) {}

		template<typename U, typename V>
		BaseVec(const BaseVec<U, 3>& xyz, const V& w) : xyz(xyz), w(static_cast<T>(w)) {}

		union
		{
			T e[4];
			struct
			{
				T x, y, z, w;
			};
			struct
			{
				T x;
				BaseVec<T, 3> yzw;
			};
			struct
			{
				BaseVec<T, 3> xyz;
				T w;
			};
			struct
			{
				BaseVec<T, 2> xy;
				BaseVec<T, 2> zw;
			};
			struct
			{
				T x;
				BaseVec<T, 2> yz;
				T w;
			};
			struct
			{
				T r, g, b, a;
			};
			struct
			{
				T r;
				BaseVec<T, 3> gba;
			};
			struct
			{
				BaseVec<T, 3> rgb;
				T a;
			};
			struct
			{
				BaseVec<T, 2> rg;
				BaseVec<T, 2> ba;
			};
			struct
			{
				T r;
				BaseVec<T, 2> gb;
				T a;
			};
			struct
			{
				T s, t, p, q;
			};
			struct
			{
				T s;
				BaseVec<T, 3> tpq;
			};
			struct
			{
				BaseVec<T, 3> stp;
				T q;
			};
			struct
			{
				BaseVec<T, 2> st;
				BaseVec<T, 2> pq;
			};
			struct
			{
				T s;
				BaseVec<T, 2> tp;
				T q;
			};
		};
	};



	/////////////////////////////////////////////////////////////////////////////////////////////
	// Generic vector class with elements of type T, and dimension D
	/////////////////////////////////////////////////////////////////////////////////////////////
	template<typename T, size_t D>
	class vec: public BaseVec<T, D>
	{
	public:
		BaseVec<T, D>::BaseVec;
		typedef T value_type;
		constexpr static const size_t D = D;


		// constructors
		vec() {}
		constexpr vec(const vec& in)
		{
			static_for<size_t, 0, D>([&](auto i) {at(i) = in.at(i); });
		}

		template<typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool> = true>
		constexpr vec(const vec<U, D>& in)
		{
			static_for<size_t, 0, D>([&](auto i)
			{
				at(i) = static_cast<T>(in.at(i));
			});
		}

		template<typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool> = true>
		constexpr vec(const U& scalar)
		{
			static_for<size_t, 0, D>([&](auto i)
			{
				at(i) = static_cast<T>(scalar);
			});
		}

		template<typename...Ts, std::enable_if_t<sizeof...(Ts) == D, bool> = true>
		constexpr vec(Ts... vals)
		{
			//static_assert(sizeof...(Ts) == D, "vec:: Cannot construct a vec of dimension D from a parameter pack of a different dimension");
			size_t i = 0;
			(void(at(i++) = static_cast<T>(vals)), ...);
		}

		template<typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool> = true>
		constexpr vec& operator=(const vec<U, D>& in);


		// accessors
		constexpr T& at(size_t i) { return BaseVec<T, D>::e[i]; }
		constexpr const T& at(size_t i) const { return BaseVec<T, D>::e[i]; }
		constexpr T& operator[](size_t i)
		{
			return at(i);
		}
		constexpr const T& operator[](size_t i) const
		{
			return at(i);
		}


		// map and zip that modify the object
		template<typename Function>
		constexpr vec& map(Function const& unaryFn);

		template<typename Function>
		constexpr vec& zip(const Function& binaryFn, const vec<T, D>& rhs);

	};


	// returns the slice [A,B)
	template<size_t A, size_t B, typename T, size_t D, std::enable_if_t< (A < B) && (B <= D), bool> = true>
	constexpr vec<T, (B - A)> slice(const vec<T, D>& in)
	{
		vec<T, B - A> res;
		static_for<size_t, A, B>([&](auto i)
		{
			res.at(i - A) = in.at(i);
		});
		return res;
	}

	// vec member funtions definitions:
	template<typename T, size_t D>
	template<typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool> >
	constexpr vec<T, D>& vec<T, D>::operator=(const vec<U, D>& in)
	{
		static_for<size_t, 0, D>([&](auto i)
		{
			at(i) = static_cast<T>(in.at(i));
		});
		return *this;
	}

	template<typename T, size_t D>
	template<typename Function>
	constexpr vec<T, D>& vec<T, D>::map(Function const& unaryFn)
	{
		static_for<size_t, 0, D>([&](auto i)
		{
			at(i) = unaryFn(at(i));
		});
		return *this;
	}

	template<typename T, size_t D>
	template<typename Function>
	constexpr vec<T, D>& vec<T, D>::zip(const Function& binaryFn, const vec<T, D>& rhs)
	{
		static_for<size_t, 0, D>([&](auto i) {at(i) = binaryFn(at(i), rhs.at(i)); });
		return *this;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////
	// Coordinate wise operations
	/////////////////////////////////////////////////////////////////////////////////////////////

	// these are outside so that it would be easier to specialize the template without much code duplication
	template<typename T, size_t D>
	constexpr vec<T, D>& operator+(vec<T, D>& in)
	{
		return in;
	}
	

	// Coordinate wise operations
	template<typename T, size_t D>
	constexpr vec<T, D>& operator+=(vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return lhs.zip(std::plus<T>(), rhs);
	}

	template<typename T, size_t D>
	constexpr vec<T, D>& operator-=(vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return lhs.zip(std::minus<T>(), rhs);
	}

	template<typename T, size_t D>
	constexpr vec<T, D>& operator*=(vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return lhs.zip(std::multiplies<T>(), rhs);
	}

	template<typename T, size_t D>
	constexpr vec<T, D>& operator/=(vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return lhs.zip(std::divides<T>(), rhs);
	}

	template<typename T, size_t D, typename U>
	vec<T, D>& operator*=(vec<T, D>& lhs, const U& scalar)
	{
		return lhs.map([&scalar](const T& x) {return x * static_cast<T>(scalar); });
	}

	template<typename T, size_t D, typename U>
	vec<T, D>& operator/=(vec<T, D>& lhs, const U& scalar)
	{
		return lhs.map([&scalar](const T& x) {return x / static_cast<T>(scalar); });
	}

	// map and zip that produce a new vector for the result

	template<typename T, size_t D, typename Function>
	constexpr vec<std::invoke_result_t<Function&, T>, D> map(Function const& unaryFn, const vec<T, D>& arg)
	{
		vec<std::invoke_result_t<Function&, T>, D> res;
		static_for<size_t, 0, D>([&](auto i) {res[i] = unaryFn(arg[i]); });
		return res;
	}

	template<typename T, size_t D, typename U = T, typename Function = T(const T&, const U&)>
	constexpr vec<std::invoke_result_t<Function&, T, U>, D> zip(const Function& binaryFn, const vec<T, D>& lhs, const vec<U, D>& rhs)
	{
		vec<std::invoke_result_t<Function&, T, U>, D> res;
		static_for<size_t, 0, D>([&](auto i) {res[i] = binaryFn(lhs[i], rhs[i]); });
		return res;
	}

	// folding functions (also known as reduce)

	template<typename T, size_t D, typename Function = T(const T&, const T&)>
	constexpr std::invoke_result_t<const Function&, T, T> foldl(Function const& binaryFn, const std::invoke_result_t<const Function&, T, T>& identity, const vec<T, D>& arg)
	{
		std::invoke_result_t<Function&, T, T> res = identity;
		static_for<size_t, 0, D>([&](auto i) {res = binaryFn(res, arg[i]); });
		return res;
	}

	template<typename T, size_t D, typename Function = T(const T&, const T&)>
	constexpr std::invoke_result_t<Function&, T, T> foldr(Function const& binaryFn, const std::invoke_result_t<const Function&, T, T>& identity, const vec<T, D>& arg)
	{
		std::invoke_result_t<Function&, T, T> res = identity;
		static_for<size_t, 0, D>([&](auto i) {res = binaryFn(arg[D - 1 - i], res); });
		return res;
	}

	/////////////////////////////////////////////////////////////////////////////////////////////
	// Coordinate wise operations
	/////////////////////////////////////////////////////////////////////////////////////////////

	template<typename T, size_t D>
	constexpr vec<T, D> operator-(const vec<T, D>& in)
	{
		return map(std::negate<T>(), in);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> operator+(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(std::plus<T>(), lhs, rhs);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> operator-(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(std::minus<T>(), lhs, rhs);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> operator*(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(std::multiplies<T>(), lhs, rhs);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> operator/(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(std::divides<T>(), lhs, rhs);
	}

	template<typename T, size_t D, typename U>
	constexpr vec<T, D> operator*(const vec<T, D>& lhs, const U& rhs)
	{
		return map([&rhs](const T& x) {return x * static_cast<T>(rhs); }, lhs);
	}

	template<typename T, size_t D, typename U>
	constexpr vec<T, D> operator/(const vec<T, D>& lhs, const U& rhs)
	{
		return map([&rhs](const T& x) {return x / static_cast<T>(rhs); }, lhs);
	}

	template<typename T, size_t D, typename U>
	constexpr vec<T, D> operator*(const U& lhs, const vec<T, D>& rhs)
	{
		return map([&lhs](const T& x) {return static_cast<T>(lhs) * x; }, rhs);
	}

	template<typename T, size_t D, typename U>
	constexpr vec<T, D> operator/(const U& lhs, const vec<T, D>& rhs)
	{
		return map([&lhs](const T& x) {return static_cast<T>(lhs) / x; }, rhs);
	}

	template<typename T, size_t D>
	constexpr vec<bool, D> operator==(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(std::equal_to<T>(), lhs, rhs);
	}

	template<typename T, size_t D>
	constexpr vec<bool, D> operator!=(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(std::not_equal_to<T>(), lhs, rhs);
	}

	template<typename T, size_t D>
	constexpr vec<bool, D> operator>(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(std::greater<T>(), lhs, rhs);
	}

	template<typename T, size_t D>
	constexpr vec<bool, D> operator>=(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(std::greater_equal<T>(), lhs, rhs);
	}

	template<typename T, size_t D>
	constexpr vec<bool, D> operator<(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(std::less<T>(), lhs, rhs);
	}

	template<typename T, size_t D>
	constexpr vec<bool, D> operator<=(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(std::less_equal<T>(), lhs, rhs);
	}

	template<size_t D>
	constexpr bool all(const vec<bool, D>& in)
	{
		return foldl(std::logical_and<bool>(), true, in);
	}

	template<size_t D>
	constexpr bool any(const vec<bool, D>& in)
	{
		return foldl(std::logical_or<bool>(), false, in);
	}

	template<typename T, size_t D>
	constexpr T dot(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return foldl(std::plus<T>(), static_cast<T>(0), lhs*rhs);
	}

	template<typename T, size_t D>
	constexpr T length_squared(const vec<T, D>& in)
	{
		return dot(in, in);
	}

	template<typename T, size_t D>
	constexpr T length(const vec<T, D>& in)
	{
		return sqrt(length_squared(in));
	}

	template<typename T, size_t D>
	constexpr vec<T, D> normalize(const vec<T, D>& in)
	{
		return in / sqrt(dot(in, in));
	}

	// coordinate wise maximum of 2 vectors
	template<typename T, size_t D>
	constexpr vec<T, D> max(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(max<T>, lhs, rhs);
	}

	// coorindate wise maximum of many vectors
	template<typename T, size_t D, typename...Ts>
	constexpr vec<T, D> max(const vec<T, D>& head, Ts...tail)
	{
		return max<T>(head, ssa::max(tail...));
	}

	// maximum component
	template<typename T, size_t D>
	constexpr T max_comp(const vec<T, D>& in)
	{
		return foldl(max<T>, in[0], in);
	}

	// returns the first index that has the maximum, ignores the rest
	template<typename T, size_t D>
	constexpr size_t max_comp_idx(const vec<T, D>& in)
	{
		size_t max_idx = 0;
		static_for<size_t, 1, D>([&](auto i) {max_idx = in[max_idx] >= in[i] ? max_idx : i; });
		return max_idx;
	}

	template<typename T, size_t D>
	constexpr vec<T, D> min(const vec<T, D>& lhs, const vec<T, D>& rhs)
	{
		return zip(max<T>, lhs, rhs);
	}

	template<typename T, size_t D>
	constexpr T min_comp(const vec<T, D>& in)
	{
		return foldl(min<T>, in[0], in);
	}

	template<typename T, size_t D>
	constexpr size_t min_comp_idx(const vec<T, D>& in)
	{
		size_t min_idx = 0;
		static_for<size_t, 1, D>([&](auto i) {min_idx = in[min_idx] <= in[i] ? min_idx : i; });
		return min_idx;
	}

	template<typename T, size_t D>
	constexpr vec<int, D> sign(const vec<T, D>& in)
	{
		return map(sign<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> abs(const vec<T, D>& in)
	{
		return map(abs<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> exp(const vec<T, D>& in)
	{
		return map(exp<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> exp2(const vec<T, D>& in)
	{
		return map(exp2<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> log(const vec<T, D>& in)
	{
		return map(log<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> log2(const vec<T, D>& in)
	{
		return map(log2<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> log10(const vec<T, D>& in)
	{
		return map(log10<T>, in);
	}

	template<typename T, size_t D, typename U>
	constexpr vec<T, D> pow(const vec<T, D>& lhs, const U& rhs)
	{
		return map([&](const T& x) {return pow<T, U>(x, rhs); }, lhs);
	}

	template<typename T, size_t D, typename U>
	constexpr vec<T, D> pow(const vec<T, D>& lhs, const vec<U, D>& rhs)
	{
		return zip(pow<T, U>, lhs, rhs);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> sqrt(const vec<T, D>& in)
	{
		return map(sqrt<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<T, D> cbrt(const vec<T, D>& in)
	{
		return map(cbrt<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<bool, D> isinf(const vec<T, D>& in)
	{
		return map(isinf<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<bool, D> isnan(const vec<T, D>& in)
	{
		return map(isnan<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<bool, D> isfinite(const vec<T, D>& in)
	{
		return map(isfinite<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<int, D> fpclassify(const vec<T, D>& in)
	{
		return map(fpclassify<T>, in);
	}

	template<typename T, size_t D>
	constexpr vec<bool, D> signbit(const vec<T, D>& in)
	{
		return map(signbit<T>, in);
	}


	template<typename T, size_t D>
	std::ostream& operator<<(std::ostream& lhs, const vec<T, D>& rhs)
	{
		static_for<size_t, 0, D>([&](auto d)
		{
			lhs << rhs[d] << "\t";
		});
		return lhs;
	}


	// glsl style shorthands

	using vec2 = vec<f32, 2>;
	using vec3 = vec<f32, 3>;
	using vec4 = vec<f32, 4>;

	using dvec2 = vec<f64, 2>;
	using dvec3 = vec<f64, 3>;
	using dvec4 = vec<f64, 4>;

	using ivec2 = vec<i32, 2>;
	using ivec3 = vec<i32, 3>;
	using ivec4 = vec<i32, 4>;

	using uvec2 = vec<u32, 2>;
	using uvec3 = vec<u32, 3>;
	using uvec4 = vec<u32, 4>;

	using bvec2 = vec<bool, 2>;
	using bvec3 = vec<bool, 3>;
	using bvec4 = vec<bool, 4>;
}