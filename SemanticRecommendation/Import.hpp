#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <armadillo>
#include <map>
#include <set>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cctype>
#include <iomanip>
#include <bitset>
#include <queue>
#include <functional>
#include <thread>  
#include <cstdlib>

using namespace std;
using namespace arma;

const double mini_factor = 1e-20;

inline
double sign(const double& x)
{
	if (x == 0)
		return 0;
	else
		return x > 0 ? +1 : -1;
}

function<double(double)> sigmoid = [](const double x) {return 1.0 / (1.0 + exp(-x)); };
function<double(double)> sigmoid_derv = [](const double x) {return sigmoid(x) * (1.0 - sigmoid(x)); };

template<typename T>
string operator - (const string& src, T& elem)
{
	stringstream ss;
	ss << src << elem;

	return ss.str();
}

template<typename T>
T& operator /(const string& src, T& elem)
{
	stringstream ss(src);
	ss >> elem;

	return elem;
}