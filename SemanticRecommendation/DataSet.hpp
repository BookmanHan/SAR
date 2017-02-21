#pragma once
#include "Import.hpp"

class DataSet
{
public:
	const string name;
	const string url;
public:
	DataSet(const string name, const string url)
		:name(name), url(url)
	{
		;
	}
};
