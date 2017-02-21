#pragma once
#include "Import.hpp"
#include "File.hpp"
#include "Utils.hpp"

class ColorPPM
{
public:
	const mat& r;
	const mat& g;
	const mat& b;

public:
	const int width;
	const int height;

public:
	ColorPPM(const mat& r, const mat& g, const mat&b)
		:r(r), g(g), b(b), width(r.n_rows), height(r.n_cols)
	{
		;
	}
};

class GrayPPM
{
public:
	const mat& lightness;

public:
	const int width;
	const int height;

public:
	GrayPPM(const mat& r)
		:lightness(r), width(r.n_rows), height(r.n_cols)
	{
		;
	}
};

#define plot(a) GrayPPM(a)

class Visualization
	:public FormatFile
{
public:
	const string file_path;

public:
	Visualization(const string file_path)
		:FormatFile(file_path, file_path, ios::binary | ios::out),
		file_path(file_path)
	{
		;
	}

	Visualization()
		:Visualization((time_namer() + ".ppm"))
	{
		;
	}

public:
	void show()
	{
		stg.close_file(name);
		system((image_reader + file_path).c_str());
	}

	void figure()
	{
		stg.close_file(name);
		system(("start " + file_path).c_str());
	}

	void refresh()
	{
		stg.close_file(name);
		stg.open_file(name, file_path, ios::binary | ios::out);
	}

	virtual void reset()
	{
		stg.open_file(name, file_path, ios::binary | ios::out);
	}
};

int operator << (Visualization& file, const mat& src)
{
	file << GrayPPM(src);
	return  0;
}

int operator << (FormatFile& file, const ColorPPM& src)
{
	try
	{
		file.try_close();
	}
	catch (exception)
	{
		;
	}

	file.reset();

	fstream& fout = file.direct();

	fout << "P6" << endl;
	fout << src.width << " " << src.height << endl;
	fout << 255 << endl;

	auto i_r = src.r.begin();
	auto i_g = src.g.begin();
	auto i_b = src.b.begin();

	while (i_r != src.r.end() && i_g != src.g.end() && i_b != src.b.end())
	{
		unsigned char r = static_cast<unsigned char>(*i_r * 255);
		unsigned char g = static_cast<unsigned char>(*i_g * 255);
		unsigned char b = static_cast<unsigned char>(*i_b * 255);

		fout.write((char*)&r, sizeof(char));
		fout.write((char*)&g, sizeof(char));
		fout.write((char*)&b, sizeof(char));

		++i_r;
		++i_g;
		++i_b;
	}

	return 3 * src.width * src.height;
}

int operator << (FormatFile& file, const GrayPPM& src)
{
	try
	{
		file.try_close();
	}
	catch (exception)
	{
		;
	}

	file.reset();

	fstream& fout = file.direct();

	fout << "P6" << endl;
	fout << src.width << " " << src.height << endl;
	fout << 255 << endl;

	auto i_pixel = src.lightness.begin();

	while (i_pixel != src.lightness.end())
	{
		unsigned char pixel = static_cast<unsigned char>(*i_pixel * 255);

		fout.write((char*)&pixel, sizeof(char));
		fout.write((char*)&pixel, sizeof(char));
		fout.write((char*)&pixel, sizeof(char));

		++i_pixel;
	}

	return src.width * src.height;
}

#define def_boxing ind_x,ind_y,x,y,size
#define def_position x,y,size

class Draw
{
public:
	mat& img;

public:
	Draw(mat& img)
		:img(img)
	{
		;
	}

public:
	bool check(int idx_x, int idx_y, int x, int y, int size)
	{
		if (idx_x < min((int)img.n_cols, x + size / 2) &&
			idx_y < min((int)img.n_rows, y + size / 2) &&
			idx_x >= max(0, x - size / 2) &&
			idx_y >= max(0, y - size / 2))
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	int left(int x, int y, int size)
	{
		return max(0, x - size / 2);
	}

	int right(int x, int y, int size)
	{
		return min((int)img.n_cols, x + size / 2) - 1;
	}

	int top(int x, int y, int size)
	{
		return max(0, y - size / 2);
	}

	int bottom(int x, int y, int size)
	{
		return min((int)img.n_rows, y + size / 2) - 1;
	}

public:
#define DrawBox &Draw::box
	void box(int x, int y, int size, double color = 0)
	{
		for (auto idx_x = max(0, x - size / 2); idx_x < min((int)img.n_cols, x + size / 2); ++idx_x)
		{
			img(idx_x, max(0, y - size / 2)) = color;
			img(idx_x, min((int)img.n_rows, y + size / 2)) = color;
		}

		for (auto idx_y = max(0, y - size / 2); idx_y < min((int)img.n_rows, y + size / 2); ++idx_y)
		{
			img(max(0, x - size / 2), idx_y) = color;
			img(min((int)img.n_cols, x + size / 2), idx_y) = color;
		}
	}

#define DrawCross &Draw::cross
	void cross(int x, int y, int size, double color = 0)
	{
		auto ind_x = (left(def_position) + right(def_position))/2;
		auto ind_y = top(def_position);
		while (check(def_boxing))
		{
			img(ind_x, ind_y) = color;
			ind_y++;
		}

		ind_x = left(def_position);
		ind_y = (top(def_position) + bottom(def_position))/2;
		while (check(def_boxing))
		{
			img(ind_x, ind_y) = color;
			ind_x++;
		}
	}

#define DrawMark &Draw::mark
	void mark(int x, int y, int size, double color = 0)
	{
		auto ind_x = left(def_position);
		auto ind_y = top(def_position);

		while (check(def_boxing))
		{
			img(ind_x, ind_y) = color;
			ind_x++;
			ind_y++;
		}

		ind_x = left(def_position);
		ind_y = bottom(def_position);
		while (check(def_boxing))
		{
			img(ind_x, ind_y) = color;
			ind_x++;
			ind_y--;
		}
	}

#define DrawCircle &Draw::circle
	void circle(int x, int y, int size, double color = 0)
	{
		auto ind_x = right(def_position);
		auto ind_y = (top(def_position) + bottom(def_position)) / 2;

		if (check(0, 0, left(def_position), top(def_position), size))
			return;
		if (check(img.n_rows, img.n_cols, right(def_position), bottom(def_position), size))
			return;

		double angle = 0;
		while (angle < 2*datum::pi)
		{
			img(x + size/2.0*cos(angle), y + size/2.0*sin(angle)) = color;
			angle += 2*datum::pi/10000.0;
		}
	}

#define DrawLine &Draw::line
	void line(int sx, int sy, int ex, int ey, double color = 0)
	{
		if (check(0, 0, sx, ex, ex - sx))
			return;
		if (check(img.n_rows, img.n_cols, sy, ey, ey - sy))
			return;

		auto ind_x = sx;
		auto ind_y = sy;

		double len = 1000;
		for (auto i = 0; i <len; ++i)
		{
			img(sx + (ex - sx)*i / len, sy + (ey - sy)*i / len) = color;
		}
	}

#define DrawUpTriangle &Draw::up_triangle
	void up_triangle(int x, int y, int size, double color = 0)
	{
		line(left(def_position), bottom(def_position), right(def_position), bottom(def_position));
		line(left(def_position), bottom(def_position), (left(def_position) + right(def_position)) / 2.0, top(def_position));
		line((left(def_position) + right(def_position)) / 2.0, top(def_position), right(def_position), bottom(def_position));
	}

#define DrawDownTriangle &Draw::down_triangle
	void down_triangle(int x, int y, int size, double color = 0)
	{
		line(left(def_position), top(def_position), right(def_position), top(def_position));
		line(left(def_position), top(def_position), (left(def_position) + right(def_position)) / 2.0, bottom(def_position));
		line((left(def_position) + right(def_position)) / 2.0, bottom(def_position), right(def_position), top(def_position));
	}

public:
	void fill(int x, int y, int size, decltype(&Draw::box) fn_draw_cortor, double color = 0)
	{
		for (auto i = 0; i <= size; ++i)
		{
			bind(fn_draw_cortor, *this, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4)(x, y, i, color);
		}
	}

	void grid(int x, int y, int size, int grid_size, decltype(&Draw::box) fn_draw_cortor, double color = 0)
	{
		for (auto i = 0; i <= size; i+=grid_size)
		{
			bind(fn_draw_cortor, *this, placeholders::_1, placeholders::_2, placeholders::_3, placeholders::_4)(x, y, i, color);
		}
	}
};

