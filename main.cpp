#include <iostream>
#include "vec.hpp"
#include "mat.hpp"

using namespace ssa;
int main()
{
	vec<int, 2> a = { 1,2 };
	mat<int, 2, 2> m = { 1,2,3,4 };
	mat<int, 2, 2> n = { vec2(2,0),vec2(0,2) };
	std::cout << m * n << "\n";
	std::cout << det(n) << "\n";
	return 0;
}