#pragma once
#include "Import.hpp"
#include "Storage.hpp"

class FormatFile
{
public:
	Storage& stg;
	const string	name;

public:
	FormatFile(
		const string name,
		const string file_path,
		const ios::ios_base::openmode open_mode,
		Storage& stg = *Storage::global_system)
		:stg(stg), name(name)
	{
		stg.open_file(name, file_path, open_mode);
	}

	FormatFile(
		const string file_path,
		const ios::ios_base::openmode open_mode,
		Storage& stg = *Storage::global_system)
		:stg(stg), name(file_path)
	{
		stg.open_file(name, file_path, open_mode);
	}

public:
	fstream& direct()
	{
		return stg.find_file(name);
	}

	void close()
	{
		stg.close_file(name);
		delete this;
	}

	void try_close()
	{
		stg.close_file(name);
	}

	virtual ~FormatFile()
	{
		;
	}

	virtual void reset()
	{
		throw exception("Not Implemented.");
	}
};

class FormatLoad
	:public FormatFile
{
protected:
	FormatLoad(const string name, const string file_path)
		:FormatFile(name, file_path, ios::binary | ios::in)
	{
		fstream& fin = direct();

		char word_size = sizeof(int);
		fin.read((char*)&word_size, sizeof(char));

		if (word_size != sizeof(int))
			throw exception("Address Model Unmatched.");
	}

	FormatLoad(const string file_path)
		:FormatFile(file_path, ios::binary | ios::in)
	{
		fstream& fin = direct();

		char word_size = sizeof(int);
		fin.read((char*)&word_size, sizeof(char));

		if (word_size != sizeof(int))
			throw exception("Address Model Unmatched.");
	}

	virtual ~FormatLoad()
	{
		;
	}

protected:
	friend FormatFile& make_fout(const string file_path);
	friend FormatFile& make_fin(const string file_path);
};

class FormatSave
	:public FormatFile
{
protected:
	FormatSave(const string name, const string file_path)
		:FormatFile(name, file_path, ios::binary | ios::out)
	{
		fstream& fout = direct();

		char word_size = sizeof(int);
		fout.write((char*)&word_size, sizeof(char));
	}

	FormatSave(const string file_path)
		:FormatFile(file_path, ios::binary | ios::out)
	{
		fstream& fout = direct();

		char word_size = sizeof(int);
		fout.write((char*)&word_size, sizeof(char));
	}

	virtual ~FormatSave()
	{
		;
	}

protected:
	friend FormatFile& make_fout(const string file_path);
	friend FormatFile& make_fin(const string file_path);
};

inline
FormatFile& make_fout(const string file_path)
{
	return *(new FormatSave(file_path));
}


inline
FormatFile& make_fin(const string file_path)
{
	return *(new FormatLoad(file_path));
}

inline
void make_close(FormatFile* that)
{
	delete that;
}

inline
void make_close(FormatFile& that)
{
	delete &that;
}

template<typename T>
inline
FormatFile& operator << (FormatFile& file, const T& src)
{
	file.direct().write((char*)&src, sizeof(T));

	return file;
}

template<typename T>
inline
FormatFile& operator >> (FormatFile& file, T& src)
{
	file.direct().read((char*)&src, sizeof(T));

	return file;
}

template<typename T>
inline
FormatFile& operator << (FormatFile& file, const Col<T>& src)
{
	fstream& fout = file.direct();

	fout.write((char*)&src.n_rows, sizeof(src.n_rows));
	fout.write((char*)&src.n_cols, sizeof(src.n_cols));
	fout.write((char*)src.memptr(), src.n_elem * sizeof(T));

	return file;
}

template<typename T>
inline
FormatFile& operator >> (FormatFile& file, Col<T>& src)
{
	arma::uword	n_row, n_col;
	fstream& fin = file.direct();

	fin.read((char*)&n_row, sizeof(n_row));
	fin.read((char*)&n_col, sizeof(n_col));

	src.resize(n_row);
	fin.read((char*)src.memptr(), n_row * n_col * sizeof(T));

	return file;
}

template<typename T>
inline
FormatFile& operator << (FormatFile& file, const Mat<T>& src)
{
	fstream& fout = file.direct();

	fout.write((char*)&src.n_rows, sizeof(src.n_rows));
	fout.write((char*)&src.n_cols, sizeof(src.n_cols));
	fout.write((char*)src.memptr(), src.n_elem * sizeof(T));

	return file;
}

template<typename T>
inline
FormatFile& operator >> (FormatFile& file, Mat<T>& src)
{
	arma::uword	n_row, n_col;
	fstream& fin = file.direct();

	fin.read((char*)&n_row, sizeof(n_row));
	fin.read((char*)&n_col, sizeof(n_col));

	src.resize(n_row, n_col);
	fin.read((char*)src.memptr(), n_row * n_col * sizeof(T));

	return file;
}

inline
FormatFile& operator << (FormatFile& file, const string& src)
{
	fstream& fout = file.direct();

	auto n_len = src.length();
	fout.write((char*)&n_len, sizeof(n_len));
	fout.write((char*)src.data(), sizeof(char)*n_len);

	return file;
}

inline
FormatFile& operator >> (FormatFile& file, string& src)
{
	fstream& fin = file.direct();

	decltype(src.length()) n_len;
	fin.read((char*)&n_len, sizeof(n_len));

	char* ca_str = new char[n_len + 1];
	fin.read((char*)ca_str, sizeof(char)*n_len);
	ca_str[n_len] = 0;
	src = ca_str;

	delete[] ca_str;

	return file;
}

template<typename T>
inline
FormatFile& operator << (FormatFile& file, const vector<T>& src)
{
	fstream& fout = file.direct();

	int size_src = src.size();
	fout.write((char*)&size_src, sizeof(int));

	for (auto i = src.begin(); i != src.end(); ++i)
	{
		file << (*i);
	}

	return file;
}

template<typename T>
inline
FormatFile& operator >> (FormatFile& file, vector<T>& src)
{
	fstream& fin = file.direct();

	int size_src = src.size();
	fin.read((char*)&size_src, sizeof(int));

	src.resize(size_src);
	for (auto i = src.begin(); i != src.end(); ++i)
	{
		file >> (*i);
	}

	return file;
}

template<typename T>
inline
FormatFile& operator << (FormatFile& file, const set<T>& src)
{
	fstream& fout = file.direct();

	int size_src = src.size();
	fout.write((char*)&size_src, sizeof(int));

	for (auto i = src.begin(); i != src.end(); ++i)
	{
		file << (*i);
	}

	return file;
}

template<typename T>
inline
FormatFile& operator >> (FormatFile& file, set<T>& src)
{
	fstream& fin = file.direct();

	int size_src = src.size();
	fin.read((char*)&size_src, sizeof(int));

	for (auto i = 0; i != size_src; ++i)
	{
		T tmp;
		file >> tmp;

		src.insert(tmp);
	}

	return file;
}

template<typename T_first, typename T_second>
inline
FormatFile& operator << (FormatFile& file, const map<T_first, T_second>& src)
{
	fstream& fout = file.direct();

	int size_src = src.size();
	fout.write((char*)&size_src, sizeof(int));

	for (auto i = src.begin(); i != src.end(); ++i)
	{
		file << i->first;
		file << i->second;
	}

	return file;
}

template<typename T_first, typename T_second>
inline
FormatFile& operator >> (FormatFile& file, map<T_first, T_second>& src)
{
	fstream& fin = file.direct();

	int size_src = src.size();
	fin.read((char*)&size_src, sizeof(int));

	for (auto i = 0; i != size_src; ++i)
	{
		pair<T_first, T_second> tmp;
		file >> tmp.first;
		file >> tmp.second;

		src.insert(tmp);
	}

	return file;
}