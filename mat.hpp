#pragma once
/*
	@author: Vassillen Chizhov 2019
	A basic matrix structure
*/
#include <type_traits>
#include "vec.hpp"
#include "typedef.hpp"

namespace ssa
{

	/*template<typename T, size_t M, size_t N>
	using mat = vec<vec<T, N>, M>;*/

	template<typename T, size_t M, size_t N = M>
	struct BaseMat : public vec<vec<T, N>, M>
	{

	};

	/*
		MxN matrix (M rows, N columns), default is a square matrix
		Each row is a vector of dimension N
	*/
	template<typename T, size_t M, size_t N = M>
	struct mat : public vec<vec<T, N>, M>
	{
		static_assert(M > 0, "mat:: Matices with 0 rows are disallowed.");
		static_assert(N > 0, "mat:: Matices with 0 columns are disallowed.");

		// use the constructors of the underlying vectors
		vec<vec<T, N>, M>::vec;
		typedef T value_type;
		typedef vec<T, N> row_type;
		typedef vec<T, M> col_type;
		typedef vec<row_type, M> base_type;
		constexpr static const size_t M = M;
		constexpr static const size_t N = N;

		constexpr row_type& at(size_t rowIdx) { return base_type::at(rowIdx); }
		constexpr const row_type& at(size_t rowIdx) const { return base_type::at(rowIdx); }
		constexpr T& at(size_t rowIdx, size_t colIdx) { return base_type::at(rowIdx).at(colIdx); }
		constexpr const T& at(size_t rowIdx, size_t colIdx) const { return base_type::at(rowIdx).at(colIdx); }

		mat() {}

		template<typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool> = true>
		mat(const vec<vec<U, N>, M>& in) : base_type(in) {}

		template<typename U, std::enable_if_t<std::is_convertible_v<U, T>, bool> = true>
		mat(const mat<U, M, N>& in) : base_type(static_cast<const vec<vec<U, N>, M>&>(in)) {}

		template<typename...Ts, std::enable_if_t<sizeof...(Ts) == (M*N), bool> = true>
		mat(const Ts&...in)
		{
			size_t i = 0;
			(void(at(i++ / M, i%M) = static_cast<T>(in)), ...);
		}

		constexpr row_type& operator[](size_t rowIdx)
		{
			return base_type::at(rowIdx);
		}

		constexpr const row_type& operator[](size_t rowIdx) const
		{
			return base_type::at(rowIdx);
		}


		constexpr T& operator()(size_t rowIdx, size_t colIdx)
		{
			return at(rowIdx, colIdx);
		}

		constexpr const T& operator()(size_t rowIdx, size_t colIdx) const
		{
			return at(rowIdx, colIdx);
		}

		// makes and returns a vector for the given column
		constexpr col_type col(size_t colIdx) const
		{
			vec<T, M> res;
			static_for<size_t, 0, M>([&](auto rowIdx) {res.at(rowIdx) = at(rowIdx, colIdx); });
			return res;
		}

	};


	// Standard binary operations
	template<typename T, size_t K, size_t L, size_t M = K, size_t N = L>
	constexpr mat<T, K, L> operator+(const mat<T, K, L>& lhs, const mat<T, M, N>& rhs)
	{
		static_assert(K == M && L == N, "operator+(mat, mat):: Matrix addition is defined only for matrices of matching dimensions.");
		return lhs + rhs;
	}

	template<typename T, size_t K, size_t L, size_t M = K, size_t N = L>
	constexpr mat<T, K, L> operator-(const mat<T, K, L>& lhs, const mat<T, M, N>& rhs)
	{
		static_assert(K == M && L == N, "operator-(mat, mat):: Matrix subtraction is defined only for matrices of matching dimensions.");
		return lhs - rhs;
	}

	// Hadamard operations
	/*template<typename T, size_t K, size_t L, size_t M = K, size_t N = L>
	constexpr mat<T, K, L> operator*(const mat<T, K, L>& lhs, const mat<T, M, N>& rhs)
	{
		static_assert(K == M && L == N, "operator*(mat, mat):: Hadamard multiplication is defined only for matrices of matching dimensions.");
		return lhs * rhs;
	}

	template<typename T, size_t K, size_t L, size_t M = K, size_t N = L>
	constexpr mat<T, K, L> operator/(const mat<T, K, L>& lhs, const mat<T, M, N>& rhs)
	{
		static_assert(K == M && L == N, "operator/(mat, mat):: Hadamard division is defined only for matrices of matching dimensions.");
		return lhs + rhs;
	}*/

	template<typename T, size_t M, size_t N>
	constexpr mat<T, N, M> transpose(const mat<T, M, N>& in)
	{
		mat<T, N, M> res;
		static_for<size_t, 0, M>([&](size_t rowIdxSrc) {
			static_for<size_t, 0, N>([&](auto colIdxSrc) {
				res.at(colIdxSrc, rowIdxSrc) = in.at(rowIdxSrc, colIdxSrc);
			});
		});
		return res;
	}

	// Multiplication by a vector on the right: M* v
	template<typename T, size_t M, size_t N, size_t K = N>
	constexpr vec<T, M> operator*(const mat<T, M, N>& lhs, const vec<T, K>& rhs)
	{
		static_assert(K == N, "operator*(mat,vec):: Matrix-vector multiplication is defined only for vectors having a dimension equal to the number of matrix columns.");
		return map([&](const vec<T, N>& lhsRow) { return dot(lhsRow, rhs); }, lhs);
	}

	// Multiplication by a vector on the left: v * M
	template<typename T, size_t M, size_t N, size_t K = M>
	constexpr vec<T, N> operator*(const vec<T, K>& lhs, const mat<T, M, N>& rhs)
	{
		static_assert(K == M, "operator*(mat,vec):: Vector-matrix multiplication is defined only for vectors having a dimension equal to the number of matrix rows.");
		//mat<T, N, M> rhsTransposed = transpose(rhs);
		//return map([&](const vec<T, M>& rhsCol) { return dot(rhsCol, lhs); }, rhsTransposed);
		return foldl(std::plus<vec<T,N>>(), vec<T,N>(0), 
			zip([&](const T& lhsCoord, const vec<T, M>& rhsRow) { return lhsCoord * rhsRow; }, lhs, rhs));
		
	}


	// Matrix multiplication between a MxK and a KxN matrix
	template<typename T, size_t M, size_t K, size_t L = K, size_t N >
	constexpr mat<T, M, N> operator*(const mat<T, M, K>& lhs, const mat<T, L, N>& rhs)
	{
		static_assert(K == L, "mat::mult:: Matrix multiplication is defined only for matrices that agree on their inner dimensions.");
		// mat<T, N, L> rhsTransposed = transpose(rhs);
		// return mat<T, M, N>(map([&](const vec<T, L>& rhsCol) { return lhs * rhsCol; }, rhsTransposed));
		return map([&](const vec<T, K>& lhsRow) {return lhsRow * rhs; }, lhs);
	}

	template<typename T, size_t N, size_t M>
	std::ostream& operator<<(std::ostream& lhs, const mat<T, N, M>& rhs)
	{
		static_for<size_t, 0, M>([&](auto i) {lhs << rhs[i] << "\n"; });
		return lhs;
	}

	// returns the reamining part of the matrix after removing row R and column N-1
	template<size_t R, typename T, size_t M, size_t N = M, std::enable_if_t<(R < M), bool> = true>
	constexpr mat<T, N - 1, M - 1> crossSliceLast(const mat<T, N, M>& in)
	{
		static_assert(N > 1 && M > 1, "mat::crossSliceLast:: Can't cut out more of the matrix");
		mat<T, N - 1, M - 1> res;
		static_for<size_t, 0, R>([&](auto i)
		{
			res[i] = slice<1, N>(in.at(i));
		});

		static_for<size_t, R + 1, M>([&](auto i)
		{
			res[i - 1] = slice<1, N>(in.at(i));
		});
		return res;
	}

	template<typename T, size_t M, size_t N = M>
	constexpr T det(const mat<T, N, M>& in)
	{
		static_assert(N == M, "mat::det Matrix determinant is defined only for square matrices.");

		T acc = 0;
		static_for<size_t, 0, M>([&](auto i)
		{
			T val = in.at(i, 0);
			T d;
			if constexpr (M == 1)
			{
				d = 1;
			}
			else
			{
				auto tempMat = crossSliceLast<i>(in);
				d = det(tempMat);
			}
			if constexpr (i.value % 2 == 0)
			{
				acc = acc + val * d;
			}
			else
			{
				acc = acc - val * d;
			}
		});
		return acc;
	}

	using mat2 = mat<f32, 2>;
	using mat3 = mat<f32, 3>;
	using mat4 = mat<f32, 4>;

	using dmat2 = mat<f64, 2>;
	using dmat3 = mat<f64, 3>;
	using dmat4 = mat<f64, 4>;

	mat4 mat4Identity()
	{
		return mat4(vec4(1, 0, 0, 0), vec4(0, 1, 0, 0), vec4(0, 0, 1, 0), vec4(0, 0, 0, 1));
	}

	mat4 mat4Translation(const vec3& v)
	{
		return mat4(vec4(1, 0, 0, v.x), vec4(0, 1, 0, v.y), vec4(0, 0, 1, v.z), vec4(0, 0, 0, 1));
	}

	mat4 mat4Scale(const f32 s)
	{
		return mat4(vec4(s, 0, 0, 0), vec4(0, s, 0, 0), vec4(0, 0, s, 0), vec4(0, 0, 0, 1));
	}

	mat4 mat4Scale(const vec3& s)
	{
		return mat4(vec4(s.x, 0, 0, 0), vec4(0, s.y, 0, 0), vec4(0, 0, s.z, 0), vec4(0, 0, 0, 1));
	}

	mat4 rotationX(f32 angle)
	{
		f32 c = cosf(angle);
		f32 s = sinf(angle);
		return mat4(vec4(1, 0, 0, 0), vec4(0, c, -s, 0), vec4(0, s, c, 0), vec4(0, 0, 0, 1));
	}

	mat4 rotationY(f32 angle)
	{
		f32 c = cosf(angle);
		f32 s = sinf(angle);
		return mat4(vec4(c, 0, s, 0), vec4(0, 1, 0, 0), vec4(-s, 0, c, 0), vec4(0, 0, 0, 1));
	}

	mat4 rotationZ(f32 angle)
	{
		f32 c = cosf(angle);
		f32 s = sinf(angle);
		return mat4(vec4(c, -s, 0, 0), vec4(s, c, 0, 0), vec4(0, 0, 1, 0), vec4(0, 0, 0, 1));
	}
}