#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <random>
#include <windows.h>

using namespace std;

wchar_t* projectPath;

Mat rgb_2_gray(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat dst(height, width, CV_8UC1);

	int i, j;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			dst.at<uchar>(i, j) = 0.299 * src.at<Vec3b>(i, j)[0] + 0.587 * src.at<Vec3b>(i, j)[1] + 0.114 * src.at<Vec3b>(i, j)[2];
		}
	}

	return dst;
}

Mat rgb_2_hsv(Mat src)
{
	int height = src.rows;
	int width = src.cols;

	Mat src_hsv(height, width, CV_8UC3);

	int i, j;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			float B = (float)src.at<Vec3b>(i, j)[0];
			float G = (float)src.at<Vec3b>(i, j)[1];
			float R = (float)src.at<Vec3b>(i, j)[2];

			float b = B / 255;
			float g = G / 255;
			float r = R / 255;

			float M = max(max(r, g), b);
			float m = min(min(r, g), b);
			float C = M - m;

			float V = M;

			float S = 0;
			if (V != 0)
			{
				S = C / V;
			}

			float H = 0;
			if (C != 0) 
			{
				if (M == r)
				{
					H = 60 * ((g - b) / C);
				}
				if (M == g)
				{
					H = 120 + 60 * ((b - r) / C);
				}
				if (M == b)
				{
					H = 240 + 60 * ((r - g) / C);
				}

				if (H < 0)
				{
					H += 360;
				}
			}

			src_hsv.at<Vec3b>(i, j)[0] = static_cast<uchar>(H / 2);
			src_hsv.at<Vec3b>(i, j)[1] = static_cast<uchar>(S * 255);
			src_hsv.at<Vec3b>(i, j)[2] = static_cast<uchar>(V * 255);
		}
	}

	return src_hsv;
}

bool is_inside(Mat img, int i, int j)
{
	int height = img.rows, width = img.cols;

	return (i >= 0 && i < height && j >= 0 && j < width);
}

void histogram(Mat src, int h[])
{
	int height = src.rows;
	int width = src.cols;
	int i, j;

	if (src.type() == CV_8UC3)
	{
		for (i = 0; i < height; ++i)
		{
			for (j = 0; j < width; ++j)
			{
				int value = (int)src.at<Vec3b>(i, j)[0];
				++h[value];
			}
		}

		return;
	}

	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			int value = (int)src.at<uchar>(i, j);
			++h[value];
		}
	}

	return;
}

const int di[] = { 0, -1, -1, -1,  0,  1, 1, 1 };
const int dj[] = { 1,  1,  0, -1, -1, -1, 0, 1 };

Mat gaussian_blur(Mat src_gray)
{
	int height = src_gray.rows;
	int width = src_gray.cols;

	Mat src_blurry(height, width, CV_8UC1);
	src_blurry = src_gray.clone();
	double gaussian_kernel[5][5] =
	{
		{ 0.000480065, 0.00500239, 0.0109262, 0.00500239, 0.000480065 },
		{ 0.00500239,  0.0521261,  0.113854,  0.0521261,  0.00500239  },
		{ 0.0109262,   0.113854,   0.24868,   0.113854,   0.0109262   },
		{ 0.00500239,  0.0521261,  0.113854,  0.0521261,  0.00500239  },
		{ 0.000480065, 0.00500239, 0.0109262, 0.00500239, 0.000480065 }
	};

	int i, j;
	for (i = 2; i < height - 2; ++i)
	{
		for (j = 2; j < width - 2; ++j)
		{
			double new_value = 0;
			int ii, jj;
			for (ii = -2; ii <= 2; ++ii)
			{
				for (jj = -2; jj <= 2; ++jj)
				{
					/*
					if (is_inside(src_gray, i + ii, j + jj))
					{
						new_value += ((double)src_gray.at<uchar>(i + ii, j + jj) * gaussian_kernel[ii + 2][jj + 2]);
					}
					*/
					new_value += ((double)src_gray.at<uchar>(i + ii, j + jj) * gaussian_kernel[ii + 2][jj + 2]);
				}
			}
			src_blurry.at<uchar>(i, j) = min(255, (int)new_value);
		}
	}

	return src_blurry;
}

struct Gradient
{
	Mat magnitude;
	Mat angles;
};

Gradient sobel(Mat src_blurry)
{
	int height = src_blurry.rows;
	int width = src_blurry.cols;

	Gradient gradient;

	Mat src_sobel(height, width, CV_8UC1, Scalar(0));
	Mat angles(height, width, CV_32FC1);
	int sobel_x[3][3] =
	{
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1 }
	};
	int sobel_y[3][3] =
	{
		{ 1, 2, 1 },
		{ 0, 0, 0 },
		{ -1, -2, -1 }
	};

	int i, j;
	for (i = 1; i < height - 1; ++i)
	{
		for (j = 1; j < width - 1; ++j)
		{
			int new_value_x = 0, new_value_y = 0;
			int ii, jj;
			for (ii = -1; ii <= 1; ++ii)
			{
				for (jj = -1; jj <= 1; ++jj)
				{
					new_value_x += ((int)src_blurry.at<uchar>(i + ii, j + jj) * sobel_x[ii + 1][jj + 1]);
					new_value_y += ((int)src_blurry.at<uchar>(i + ii, j + jj) * sobel_y[ii + 1][jj + 1]);
				}
			}
			//int sum = abs(new_value_x) + abs(new_value_y);
			int sum = (int)sqrt(pow(new_value_x, 2) + pow(new_value_y, 2)) / (int)(4 * sqrt(2));
			src_sobel.at<uchar>(i, j) = min(255, sum);

			angles.at<float>(i, j) = (float)atan2(new_value_y, new_value_x) * 180.0 / PI + 180.0;
			/*
			if (angles.at<float>(i, j) < 0)
			{
				angles.at<float>(i, j) += 180.0;
			}
			*/
		}
	}

	gradient.magnitude = src_sobel;
	gradient.angles = angles;
	return gradient;
}

void non_maximum_suppression(Gradient& gradient)
{
	Mat src_sobel = gradient.magnitude;
	int height = src_sobel.rows;
	int width = src_sobel.cols;

	int i, j;
	for (i = 1; i < height - 1; ++i)
	{
		for (j = 1; j < width - 1; ++j)
		{
			float current_angle = gradient.angles.at<float>(i, j);

			if ((current_angle <= 22.5) ||
				(current_angle > 337.5) ||
				(current_angle > 157.5 && current_angle <= 202.5))
			{
				if (!((int)src_sobel.at<uchar>(i, j) > (int)src_sobel.at<uchar>(i, j - 1) && (int)src_sobel.at<uchar>(i, j) > (int)src_sobel.at<uchar>(i, j + 1)))
				{
					src_sobel.at<uchar>(i, j) = 0;
				}
			}
			else if ((current_angle > 22.5 && current_angle <= 67.5) ||
					 (current_angle > 202.5 && current_angle <= 247.5))
			{
				if (!((int)src_sobel.at<uchar>(i, j) > (int)src_sobel.at<uchar>(i - 1, j + 1) && (int)src_sobel.at<uchar>(i, j) > (int)src_sobel.at<uchar>(i + 1, j - 1)))
				{
					src_sobel.at<uchar>(i, j) = 0;
				}
			}
			else if ((current_angle > 67.5 && current_angle <= 112.5) ||
					 (current_angle > 247.5 && current_angle <= 292.5))
			{
				if (!((int)src_sobel.at<uchar>(i, j) > (int)src_sobel.at<uchar>(i - 1, j) && (int)src_sobel.at<uchar>(i, j) > (int)src_sobel.at<uchar>(i + 1, j)))
				{
					src_sobel.at<uchar>(i, j) = 0;
				}
			}
			else
			{
				if (!((int)src_sobel.at<uchar>(i, j) > (int)src_sobel.at<uchar>(i - 1, j - 1) && (int)src_sobel.at<uchar>(i, j) > (int)src_sobel.at<uchar>(i + 1, j + 1)))
				{
					src_sobel.at<uchar>(i, j) = 0;
				}
			}
		}
	}

	return;
}

void double_th_hysteresis(Gradient& gradient, int adaptive_th, float k)
{
	Mat src_sobel = gradient.magnitude;
	int height = src_sobel.rows;
	int width = src_sobel.cols;

	int i, j;
	int upper_th = adaptive_th, lower_th = adaptive_th * k;
	for (i = 1; i < height - 1; ++i)
	{
		for (j = 1; j < width - 1; ++j)
		{
			if ((int)src_sobel.at<uchar>(i, j) < lower_th)
			{
				src_sobel.at<uchar>(i, j) = 0; // discarded
			}
			else if ((int)src_sobel.at<uchar>(i, j) > upper_th)
			{
				src_sobel.at<uchar>(i, j) = 255; // strong edge
			}
			else
			{
				src_sobel.at<uchar>(i, j) = 128; // weak edge
			}
		}
	}
	
	int h;
	for (i = 1; i < height - 1; ++i)
	{
		for (j = 1; j < width - 1; ++j)
		{
			if ((int)src_sobel.at<uchar>(i, j) == 255)
			{
				queue<pair<int, int>> q;
				q.push({ i, j });
				while (!q.empty())
				{
					pair<int, int> curr_pair = q.front();
					q.pop();
					for (h = 0; h < 8; ++h)
					{
						int new_i = curr_pair.first + di[h];
						int new_j = curr_pair.second + dj[h];
						if ((int)src_sobel.at<uchar>(new_i, new_j) == 128)
						{
							src_sobel.at<uchar>(new_i, new_j) = 255;
							q.push({ new_i, new_j });
						}
					}
				}
			}
		}
	}

	for (i = 1; i < height - 1; ++i)
	{
		for (j = 1; j < width - 1; ++j)
		{
			if ((int)src_sobel.at<uchar>(i, j) == 128)
			{
				src_sobel.at<uchar>(i, j) = 0;
			}
		}
	}

	return;
}

Mat canny(Mat src_color)
{
	int height = src_color.rows;
	int width = src_color.cols;

	Mat src_gray = rgb_2_gray(src_color);
	//imwrite("example/grayscale.png", src_gray);

	Mat src_blurry = gaussian_blur(src_gray);
	//imwrite("example/gauss blur.png", src_blurry);

	Gradient gradient = sobel(src_blurry);
	//imwrite("example/sobel - magnitude.png", gradient.magnitude);

	non_maximum_suppression(gradient);
	//imwrite("example/nms.png", gradient.magnitude);

	int h[256] = { 0 };
	histogram(gradient.magnitude, h);
	float p = 0.1;
	int number_of_non_edge_pixels = (1 - p) * (height * width - h[0]);
	int i, sum = 0, adaptive_th = 1;
	for (i = 1; i < 256; ++i)
	{
		sum += h[i];
		if (sum > number_of_non_edge_pixels)
		{
			adaptive_th = i;
			break;
		}
	}

	double_th_hysteresis(gradient, adaptive_th, 0.4);

	return gradient.magnitude;
}

struct Vec3iComparator
{
	bool operator()(Vec3i lhs, Vec3i rhs) const
	{
		return ((lhs[0] < rhs[0]) || (lhs[0] == rhs[0] && lhs[1] < rhs[1]) || (lhs[0] == rhs[0] && lhs[1] == rhs[1] && lhs[2] < rhs[2]));
	}
};

bool is_inside_circle(int x, int y, int radius, int x_circle, int y_circle, int radius_circle, vector<double> sin_v, vector<double> cos_v)
{
	if ((pow(x - x_circle, 2) + pow(y - y_circle, 2) <= pow(radius_circle, 2)))
	{
		return true;
	}

	int d;
	for (d = 0; d < 360; ++d)
	{
		int a = x - radius * cos_v[d];
		int b = y - radius * sin_v[d];
		if ((pow(a - x_circle, 2) + pow(b - y_circle, 2) <= pow(radius_circle, 2)))
		{
			return true;
		}
	}

	return false;
}

void delete_redundant_circles(vector<Vec3i>& circles, Mat accumulator, int minimum_radius, vector<double> sin_v, vector<double> cos_v)
{
	set<Vec3i, Vec3iComparator> circles_to_be_removed;
	int n = circles.size();
	int i, j;
	for (i = 0; i < n - 1; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{
			int x_i = circles[i][0], y_i = circles[i][1], r_i = circles[i][2];
			int x_j = circles[j][0], y_j = circles[j][1], r_j = circles[j][2];
			if (is_inside_circle(x_j, y_j, r_j, x_i, y_i, r_i, sin_v, cos_v) &&
				accumulator.at<int>(x_j, y_j, r_j - minimum_radius) <= accumulator.at<int>(x_i, y_i, r_i - minimum_radius))
			{
				circles_to_be_removed.insert(circles[j]);
			}
			else if (is_inside_circle(x_i, y_i, r_i, x_j, y_j, r_j, sin_v, cos_v) &&
					 accumulator.at<int>(x_i, y_i, r_i - minimum_radius) <= accumulator.at<int>(x_j, y_j, r_j - minimum_radius))
			{
				circles_to_be_removed.insert(circles[i]);
			}
		}
	}

	set<Vec3i>::iterator it;
	for (it = circles_to_be_removed.begin(); it != circles_to_be_removed.end(); ++it)
	{
		Vec3i current_circle_to_be_removed = (*it);
		for (i = n - 1; i >= 0; --i)
		{
			if (circles[i] == current_circle_to_be_removed)
			{
				for (j = i; j < n - 1; ++j)
				{
					circles[j] = circles[j + 1];
				}
				--n;
			}
		}
	}
	circles.resize(n);

	return;
}

void sort_circles(vector<Vec3i>& circles, Mat accumulator, int minimum_radius)
{
	int n = circles.size(), i, j;
	for (i = 0; i < n - 1; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{
			int x_i = circles[i][0], y_i = circles[i][1], r_i = circles[i][2];
			int x_j = circles[j][0], y_j = circles[j][1], r_j = circles[j][2];
			if (accumulator.at<int>(x_i, y_i, r_i - minimum_radius) < accumulator.at<int>(x_j, y_j, r_j - minimum_radius))
			{
				swap(circles[i], circles[j]);
			}
		}
	}

	return;
}

vector<Vec3i> hough_circle(Mat src_canny)
{
	int height = src_canny.rows, width = src_canny.cols;
	int minimum_radius = 15, maximum_radius = 150;
	int depth = maximum_radius - minimum_radius + 1;

	int sizes[3] = { height, width, depth };
	Mat accumulator(sizeof(sizes) / sizeof(sizes[0]), sizes, CV_32SC1, Scalar(0));

	// "look-up tables" for sin and cos
	vector<double> sin_v, cos_v;
	int d;
	for (d = 0; d < 360; ++d)
	{
		double radians = (double)d * PI / 180.0;

		sin_v.push_back(sin(radians));
		cos_v.push_back(cos(radians));
	}

	int i, j, r;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			if ((int)src_canny.at<uchar>(i, j) == 255)
			{
				for (r = minimum_radius; r <= maximum_radius; ++r)
				{
					for (d = 0; d < 360; ++d)
					{
						int a = i - r * cos_v[d];
						int b = j - r * sin_v[d];
						if (is_inside(src_canny, a, b))
						{
							++accumulator.at<int>(a, b, r - minimum_radius);
						}
					}
				}
			}
		}
	}

	int threshold = 120;
	vector<Vec3i> circles;
	int index = -1, h;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			for (r = maximum_radius; r >= minimum_radius; --r)
			{
				if (accumulator.at<int>(i, j, r - minimum_radius) > threshold)
				{
					if (index == -1 &&
						(is_inside(src_canny, i - r, j) && is_inside(src_canny, i + r, j) && is_inside(src_canny, i, j - r) && is_inside(src_canny, i, j + r)))
					{
						circles.push_back(Vec3i(i, j, r));
						++index;

						break;
					}

					bool fail = false;
					for (h = 0; h <= index && !fail; ++h)
					{
						int x_h = circles[h][0], y_h = circles[h][1], r_h = circles[h][2];
						if (is_inside_circle(i, j, r, x_h, y_h, r_h, sin_v, cos_v) &&
							accumulator.at<int>(i, j, r - minimum_radius) <= accumulator.at<int>(x_h, y_h, r_h - minimum_radius))
						{
							fail = true;
						}
					}

					if (!fail &&
						(is_inside(src_canny, i - r, j) && is_inside(src_canny, i + r, j) && is_inside(src_canny, i, j - r) && is_inside(src_canny, i, j + r)))
					{
						circles.push_back(Vec3i(i, j, r));
						++index;

						break;
					}
				}
			}
		}
	}

	int number_of_remaining_circles = 30;
	sort_circles(circles, accumulator, minimum_radius);
	if (circles.size() > number_of_remaining_circles)
	{
		circles.resize(number_of_remaining_circles);
	}
	delete_redundant_circles(circles, accumulator, minimum_radius, sin_v, cos_v);

	return circles;
}

bool is_inside_square(int x_min_1, int x_max_1, int y_min_1, int y_max_1, int x_min_2, int x_max_2, int y_min_2, int y_max_2)
{
	return ((x_min_2 <= x_min_1 && x_min_1 <= x_max_2 && y_min_2 <= y_min_1 && y_min_1 <= y_max_2) ||
			(x_min_2 <= x_min_1 && x_min_1 <= x_max_2 && y_min_2 <= y_max_1 && y_max_1 <= y_max_2) ||
			(x_min_2 <= x_max_1 && x_max_1 <= x_max_2 && y_min_2 <= y_min_1 && y_min_1 <= y_max_2) ||
			(x_min_2 <= x_max_1 && x_max_1 <= x_max_2 && y_min_2 <= y_max_1 && y_max_1 <= y_max_2));
}

bool is_center_inside_square(int x, int y, int x_min_1, int x_max_1, int y_min_1, int y_max_1)
{
	return (x_min_1 <= x && x <= x_max_1 && y_min_1 <= y && y <= y_max_1);
}

void delete_redundant_squares(vector<Vec3i>& squares, Mat accumulator, int minimum_length)
{
	set<Vec3i, Vec3iComparator> squares_to_be_removed;
	int n = squares.size();
	int i, j;
	for (i = 0; i < n - 1; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{
			int x_i = squares[i][1], y_i = squares[i][0], l_i = squares[i][2];
			int x_j = squares[j][1], y_j = squares[j][0], l_j = squares[j][2];
			if ((is_center_inside_square(x_j, y_j, x_i - l_i, x_i + l_i, y_i - l_i, y_i + l_i) || is_inside_square(x_j - l_j, x_j + l_j, y_j - l_j, y_j + l_j, x_i - l_i, x_i + l_i, y_i - l_i, y_i + l_i)) &&
				accumulator.at<float>(y_j, x_j, l_j - minimum_length) <= accumulator.at<float>(y_i, x_i, l_i - minimum_length))
			{
				squares_to_be_removed.insert(squares[j]);
			}
			else if ((is_center_inside_square(x_i, y_i, x_j - l_j, x_j + l_j, y_j - l_j, y_j + l_j) || is_inside_square(x_i - l_i, x_i + l_i, y_i - l_i, y_i + l_i, x_j - l_j, x_j + l_j, y_j - l_j, y_j + l_j)) &&
					 accumulator.at<float>(y_i, x_i, l_i - minimum_length) <= accumulator.at<float>(y_j, x_j, l_j - minimum_length))
			{
				squares_to_be_removed.insert(squares[i]);
			}
		}
	}

	set<Vec3i>::iterator it;
	for (it = squares_to_be_removed.begin(); it != squares_to_be_removed.end(); ++it)
	{
		Vec3i current_square_to_be_removed = (*it);
		for (i = n - 1; i >= 0; --i)
		{
			if (squares[i] == current_square_to_be_removed)
			{
				for (j = i; j < n - 1; ++j)
				{
					squares[j] = squares[j + 1];
				}
				--n;
			}
		}
	}
	squares.resize(n);

	return;
}

void sort_squares(vector<Vec3i>& squares, Mat accumulator, int minimum_length)
{
	int n = squares.size(), i, j;
	for (i = 0; i < n - 1; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{
			int x_i = squares[i][1], y_i = squares[i][0], l_i = squares[i][2];
			int x_j = squares[j][1], y_j = squares[j][0], l_j = squares[j][2];
			if (accumulator.at<float>(y_i, x_i, l_i - minimum_length) < accumulator.at<float>(y_j, x_j, l_j - minimum_length))
			{
				swap(squares[i], squares[j]);
			}
		}
	}

	return;
}

vector<Vec3i> hough_square(Mat src_canny)
{
	int height = src_canny.rows, width = src_canny.cols;
	int minimum_length = 20, maximum_length = 75; // actually, these are the halfs of the lengths
	int depth = maximum_length - minimum_length + 1;

	int sizes[3] = { height, width, depth };
	Mat accumulator(sizeof(sizes) / sizeof(sizes[0]), sizes, CV_32FC1, Scalar(0.0));

	int i, j, l;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			if ((int)src_canny.at<uchar>(i, j) == 255)
			{
				for (l = minimum_length; l <= maximum_length; ++l)
				{
					float perimeter = 1.0 / (8.0 * (float)l);

					int ii;
					for (ii = i - l; ii <= i + l; ++ii)
					{
						if (is_inside(src_canny, ii, j - l))
						{
							accumulator.at<float>(ii, j - l, l - minimum_length) += perimeter;
						}
						if (is_inside(src_canny, ii, j + l))
						{
							accumulator.at<float>(ii, j + l, l - minimum_length) += perimeter;
						}
					}

					int jj;
					for (jj = j - l; jj <= j + l; ++jj)
					{
						if (is_inside(src_canny, i - l, jj))
						{
							accumulator.at<float>(i - l, jj, l - minimum_length) += perimeter;
						}
						if (is_inside(src_canny, i + l, jj))
						{
							accumulator.at<float>(i + l, jj, l - minimum_length) += perimeter;
						}
					}
				}
			}
		}
	}

	float threshold = 0.1;
	vector<Vec3i> squares;
	int index = -1, h;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			for (l = maximum_length; l >= minimum_length; --l)
			{
				if (accumulator.at<float>(i, j, l - minimum_length) > threshold)
				{
					if (index == -1 &&
						(is_inside(src_canny, i - l, j - l) && is_inside(src_canny, i + l, j - l) && is_inside(src_canny, i + l, j + l) && is_inside(src_canny, i - l, j + l)))
					{
						squares.push_back(Vec3i(i, j, l));
						++index;

						break;
					}

					bool fail = false;
					for (h = 0; h <= index && !fail; ++h)
					{
						int x_h = squares[h][1], y_h = squares[h][0], l_h = squares[h][2];
						if (is_inside_square(j - l, j + l, i - l, i + l, x_h - l_h, x_h + l_h, y_h - l_h, y_h + l_h) &&
							accumulator.at<float>(i, j, l - minimum_length) <= accumulator.at<float>(y_h, x_h, l_h - minimum_length))
						{
							fail = true;
						}
					}

					if (!fail &&
						(is_inside(src_canny, i - l, j - l) && is_inside(src_canny, i + l, j - l) && is_inside(src_canny, i + l, j + l) && is_inside(src_canny, i - l, j + l)))
					{
						squares.push_back(Vec3i(i, j, l));
						++index;

						break;
					}
				}
			}
		}
	}

	int number_of_remaining_squares = 30;
	sort_squares(squares, accumulator, minimum_length);
	if (squares.size() > number_of_remaining_squares)
	{
		squares.resize(number_of_remaining_squares);
	}
	delete_redundant_squares(squares, accumulator, minimum_length);

	return squares;
}

struct Gradient2
{
	Mat magnitude_x;
	Mat magnitude_y;
};

Gradient2 sobel_harris(Mat src_gray)
{
	int height = src_gray.rows;
	int width = src_gray.cols;

	Gradient2 gradient;

	Mat src_sobel_x(height, width, CV_32FC1, Scalar(0));
	Mat src_sobel_y(height, width, CV_32FC1, Scalar(0));
	float sobel_x[3][3] =
	{
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1 }
	};
	float sobel_y[3][3] =
	{
		{ 1, 2, 1 },
		{ 0, 0, 0 },
		{ -1, -2, -1 }
	};

	int i, j;
	for (i = 1; i < height - 1; ++i)
	{
		for (j = 1; j < width - 1; ++j)
		{
			float new_value_x = 0, new_value_y = 0;
			int ii, jj;
			for (ii = -1; ii <= 1; ++ii)
			{
				for (jj = -1; jj <= 1; ++jj)
				{
					new_value_x += ((float)src_gray.at<uchar>(i + ii, j + jj) * sobel_x[ii + 1][jj + 1]);
					new_value_y += ((float)src_gray.at<uchar>(i + ii, j + jj) * sobel_y[ii + 1][jj + 1]);
				}
			}
			src_sobel_x.at<float>(i, j) = new_value_x / (4 * sqrt(2));
			src_sobel_y.at<float>(i, j) = new_value_y / (4 * sqrt(2));
		}
	}

	gradient.magnitude_x = src_sobel_x;
	gradient.magnitude_y = src_sobel_y;
	return gradient;
}

Mat gaussian_blur_harris(Mat src_sobel)
{
	int height = src_sobel.rows;
	int width = src_sobel.cols;

	Mat src_blurry(height, width, CV_32FC1);
	src_blurry = src_sobel.clone();
	double gaussian_kernel[5][5] =
	{
		{ 0.000480065, 0.00500239, 0.0109262, 0.00500239, 0.000480065 },
		{ 0.00500239,  0.0521261,  0.113854,  0.0521261,  0.00500239  },
		{ 0.0109262,   0.113854,   0.24868,   0.113854,   0.0109262   },
		{ 0.00500239,  0.0521261,  0.113854,  0.0521261,  0.00500239  },
		{ 0.000480065, 0.00500239, 0.0109262, 0.00500239, 0.000480065 }
	};

	int i, j;
	for (i = 2; i < height - 2; ++i)
	{
		for (j = 2; j < width - 2; ++j)
		{
			float new_value = 0;
			int ii, jj;
			for (ii = -2; ii <= 2; ++ii)
			{
				for (jj = -2; jj <= 2; ++jj)
				{
					new_value += (src_sobel.at<float>(i + ii, j + jj) * gaussian_kernel[ii + 2][jj + 2]);
				}
			}
			src_blurry.at<float>(i, j) = new_value;
		}
	}

	return src_blurry;
}

struct Triangle
{
	Point A, B, C;
	int votes;
};

float dist(Point A, Point B)
{
	return (float)sqrt(pow(A.x - B.x, 2) + pow(A.y - B.y, 2));
}

bool is_between(float l1, float l2, float margin_of_error)
{
	return (l1 - margin_of_error <= l2 && l2 <= l1 + margin_of_error);
}

bool is_inside_triangle(Triangle t1, Triangle t2)
{
	Point A1 = t1.A;
	Point B1 = t1.B;
	Point C1 = t1.C;
	Point O1 = (A1 + B1 + C1) / 3;

	Point A2 = t2.A;
	Point B2 = t2.B;
	Point C2 = t2.C;

	int max_x = max(max(A2.x, B2.x), C2.x);
	int min_x = min(min(A2.x, B2.x), C2.x);
	int max_y = max(max(A2.y, B2.y), C2.y);
	int min_y = min(min(A2.y, B2.y), C2.y);

	return ((min_x <= A1.x && A1.x <= max_x && min_y <= A1.y && A1.y <= max_y) ||
		    (min_x <= B1.x && B1.x <= max_x && min_y <= B1.y && B1.y <= max_y) ||
		    (min_x <= C1.x && C1.x <= max_x && min_y <= C1.y && C1.y <= max_y) ||
			(min_x <= O1.x && O1.x <= max_x && min_y <= O1.y && O1.y <= max_y));
}

void delete_redundant_triangles(vector<Triangle>& triangles)
{
	vector<Triangle> triangles_to_be_removed;
	int n = triangles.size(), i, j;
	for (i = 0; i < n - 1; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{
			int votes_i = triangles[i].votes;
			int votes_j = triangles[j].votes;

			if (votes_i > votes_j && is_inside_triangle(triangles[i], triangles[j]))
			{
				triangles_to_be_removed.push_back(triangles[i]);
			}
			else if (votes_j > votes_i && is_inside_triangle(triangles[j], triangles[i]))
			{
				triangles_to_be_removed.push_back(triangles[j]);
			}
		}
	}

	vector<Triangle>::iterator it;
	for (it = triangles_to_be_removed.begin(); it != triangles_to_be_removed.end(); ++it)
	{
		Triangle current_triangle_to_be_removed = (*it);
		for (i = n - 1; i >= 0; --i)
		{
			if (triangles[i].votes == current_triangle_to_be_removed.votes && 
				triangles[i].A == current_triangle_to_be_removed.A &&
				triangles[i].B == current_triangle_to_be_removed.B &&
				triangles[i].C == current_triangle_to_be_removed.C)
			{
				for (j = i; j < n - 1; ++j)
				{
					triangles[j] = triangles[j + 1];
				}
				--n;
			}
		}
	}
	triangles.resize(n);

	for (i = 0; i < n; ++i)
	{
		for (j = n - 1; j >= 0; --j)
		{
			if (i != j)
			{
				int votes_i = triangles[i].votes;
				int votes_j = triangles[j].votes;

				if (votes_i == votes_j && is_inside_triangle(triangles[j], triangles[i]))
				{
					int k;
					for (k = j; k < n - 1; ++k)
					{
						triangles[k] = triangles[k + 1];
					}
					--n;
				}
			}
		}
	}
	triangles.resize(n);

	return;
}

void sort_triangles(vector<Triangle>& triangles)
{
	int n = triangles.size(), i, j;
	for (i = 0; i < n - 1; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{
			if (triangles[i].votes > triangles[j].votes)
			{
				swap(triangles[i], triangles[j]);
			}
		}
	}

	return;
}

vector<Triangle> harris_corner_detection_triangles(Mat src_approx_contours, Mat approx_contours_points)
{
	double k = 0.04;

	int height = src_approx_contours.rows;
	int width = src_approx_contours.cols;

	Gradient2 gradient = sobel_harris(src_approx_contours);

	Mat sobel_x2 = gradient.magnitude_x.mul(gradient.magnitude_x);
	Mat sobel_y2 = gradient.magnitude_y.mul(gradient.magnitude_y);
	Mat sobel_xy = gradient.magnitude_x.mul(gradient.magnitude_y);

	Mat sobel_x2_blur = gaussian_blur_harris(sobel_x2);
	Mat sobel_y2_blur = gaussian_blur_harris(sobel_y2);
	Mat sobel_xy_blur = gaussian_blur_harris(sobel_xy);

	Mat x2y2 = sobel_x2_blur.mul(sobel_y2_blur);
	Mat xy = sobel_xy_blur.mul(sobel_xy_blur);
	Mat determinant = x2y2 - xy;
	Mat trace2 = (sobel_x2_blur + sobel_y2_blur).mul((sobel_x2_blur + sobel_y2_blur));

	Mat dst = determinant - k * trace2;

	vector<Point> corners;
	int window_size = 3;
	int i, j;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			if (approx_contours_points.at<uchar>(i, j) == 255)
			{
				float maximum_value = 0;
				Point corner;
				int ii, jj;
				for (ii = -window_size; ii <= window_size; ++ii)
				{
					for (jj = -window_size; jj <= window_size; ++jj)
					{
						if (is_inside(dst, i + ii, j + jj) && dst.at<float>(i + ii, j + jj) > maximum_value)
						{
							maximum_value = dst.at<float>(i + ii, j + jj);
							corner = { j + jj, i + ii };
						}
					}
				}

				if (maximum_value != 0)
				{
					corners.push_back(corner);
				}
			}
		}
	}

	int minimum_length = 30, maximum_length = 100;
	vector<Triangle> triangles;
	int margin_of_error_length = 6;
	int margin_of_error_base = 4;
	int h, n = corners.size();
	for (i = 0; i < n - 2; ++i)
	{
		for (j = i + 1; j < n - 1; ++j)
		{
			for (h = j + 1; h < n; ++h)
			{
				Point A = corners[i];
				Point B = corners[j];
				Point C = corners[h];

				float AB = dist(A, B);
				float AC = dist(A, C);
				float BC = dist(B, C);

				// we need to have a point which is higher than the other two in terms of y, while the other two need to be somewhat on the same y
				int ys[3] = { A.y, B.y, C.y };
				sort(ys, ys + 2);

				if ((AB > minimum_length && AC > minimum_length && BC > minimum_length && AB < maximum_length && AC < maximum_length && BC < maximum_length) &&
					(ys[1] - margin_of_error_base < ys[2] && ys[2] < ys[1] + margin_of_error_base) &&
					(ys[2] - margin_of_error_base < ys[1] && ys[1] < ys[2] + margin_of_error_base) &&
					(is_between(AB, AC, margin_of_error_length)) &&
					(is_between(AB, BC, margin_of_error_length)) &&
					(is_between(AC, AB, margin_of_error_length)) &&
					(is_between(AC, BC, margin_of_error_length)) &&
					(is_between(BC, AB, margin_of_error_length)) &&
					(is_between(BC, AC, margin_of_error_length)))
				{
					int max_x = max(max(A.x, B.x), C.x);
					int min_x = min(min(A.x, B.x), C.x);
					int max_y = max(max(A.y, B.y), C.y);
					int min_y = min(min(A.y, B.y), C.y);

					int votes = 0;
					int x, y;
					for (x = min_x; x <= max_x; ++x)
					{
						for (y = min_y; y <= max_y; ++y)
						{
							if (approx_contours_points.at<uchar>(y, x) == 255)
							{
								++votes;
							}
						}
					}

					Triangle current_triangle = { A, B, C, votes };
					int nn = triangles.size(), ii;
					bool fail = false;
					for (ii = 0; ii < nn; ++ii)
					{
						if (current_triangle.votes > triangles[ii].votes && is_inside_triangle(current_triangle, triangles[ii]))
						{
							fail = true;
						}
					}
					if (!fail)
					{
						triangles.push_back(current_triangle);
					}
				}
			}
		}
	}

	int number_of_remaining_triangles = 30;
	sort_triangles(triangles);
	if (triangles.size() > number_of_remaining_triangles)
	{
		triangles.resize(number_of_remaining_triangles);
	}
	delete_redundant_triangles(triangles);

	return triangles;
}

Mat in_range(Mat src_hsv, Scalar lower_bound, Scalar upper_bound)
{
	int height = src_hsv.rows, width = src_hsv.cols;
	Mat mask(height, width, CV_8UC1, Scalar(0));

	int i, j;
	for (i = 0; i < height; ++i) 
	{
		for (j = 0; j < width; ++j) 
		{
			Vec3b src_hsv_pixel = src_hsv.at<Vec3b>(i, j);
			if ((src_hsv_pixel[0] >= lower_bound[0] && src_hsv_pixel[0] <= upper_bound[0]) &&
				(src_hsv_pixel[1] >= lower_bound[1] && src_hsv_pixel[1] <= upper_bound[1]) &&
				(src_hsv_pixel[2] >= lower_bound[2] && src_hsv_pixel[2] <= upper_bound[2])) 
			{
				mask.at<uchar>(i, j) = 255;
			}
		}
	}

	return mask;
}

Mat bitwise_or(Mat mask_1, Mat mask_2)
{
	int height = mask_1.rows, width = mask_1.cols;
	Mat mask_f(height, width, CV_8UC1, Scalar(0));

	int i, j;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			mask_f.at<uchar>(i, j) = mask_1.at<uchar>(i, j) | mask_2.at<uchar>(i, j);
		}
	}

	return mask_f;
}

Mat bitwise_and(Mat src_color, Mat mask)
{
	int height = src_color.rows, width = src_color.cols;
	Mat extracted_color(height, width, CV_8UC3, Scalar(0, 0, 0));

	int i, j;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			extracted_color.at<Vec3b>(i, j)[0] = src_color.at<Vec3b>(i, j)[0] & mask.at<uchar>(i, j);
			extracted_color.at<Vec3b>(i, j)[1] = src_color.at<Vec3b>(i, j)[1] & mask.at<uchar>(i, j);
			extracted_color.at<Vec3b>(i, j)[2] = src_color.at<Vec3b>(i, j)[2] & mask.at<uchar>(i, j);
		}
	}

	return extracted_color;
}

vector<Mat> filter_colors(Mat src_color, Mat src_hsv)
{
	Scalar red_lower_bound_1 = Scalar(0, 30, 30);
	Scalar red_upper_bound_1 = Scalar(10, 255, 255);
	Scalar red_lower_bound_2 = Scalar(160, 30, 30);
	Scalar red_upper_bound_2 = Scalar(180, 255, 255);

	Scalar blue_lower_bound = Scalar(100, 50, 50);
	Scalar blue_upper_bound = Scalar(130, 255, 255);

	Mat red_mask_1 = in_range(src_hsv, red_lower_bound_1, red_upper_bound_1);
	Mat red_mask_2 = in_range(src_hsv, red_lower_bound_2, red_upper_bound_2);
	Mat red_mask_f = bitwise_or(red_mask_1, red_mask_2);

	Mat blue_mask = in_range(src_hsv, blue_lower_bound, blue_upper_bound);

	Mat src_intermediate_red = bitwise_and(src_color, red_mask_f);
	Mat src_intermediate_blue = bitwise_and(src_color, blue_mask);

	int height = src_color.rows, width = src_color.cols;
	Mat src_filtered(height, width, CV_8UC3, Scalar(0, 0, 0));
	Mat src_filtered_red(height, width, CV_8UC3, Scalar(0, 0, 0));
	Mat src_filtered_blue(height, width, CV_8UC3, Scalar(0, 0, 0));

	int i, j;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			Vec3b color_red = src_intermediate_red.at<Vec3b>(i, j);
			Vec3b color_blue = src_intermediate_blue.at<Vec3b>(i, j);
			src_filtered.at<Vec3b>(i, j) = color_red + color_blue;
			src_filtered_red.at<Vec3b>(i, j) = color_red;
			src_filtered_blue.at<Vec3b>(i, j) = color_blue;
		}
	}

	return { src_filtered, src_filtered_red, src_filtered_blue };
}

/*
Mat equalize_histogram(Mat src) 
{
	int height = src.rows, width = src.cols;

	Mat ycrcb;
	cvtColor(src, ycrcb, COLOR_BGR2YCrCb);

	int hist[256] = { 0 };
	histogram(ycrcb, hist);

	int cdf[256] = { 0 };
	cdf[0] = hist[0];
	int i;
	for (i = 1; i < 256; ++i) 
	{
		cdf[i] = cdf[i - 1] + hist[i];
	}

	for (i = 0; i < 256; ++i) 
	{
		cdf[i] = (cdf[i] * 255) / (height * width);
	}

	int j;
	for (i = 0; i < height; ++i) 
	{
		for (j = 0; j < width; ++j) 
		{
			ycrcb.at<Vec3b>(i, j)[0] = cdf[ycrcb.at<Vec3b>(i, j)[0]];
		}
	}

	Mat src_2;
	cvtColor(ycrcb, src_2, COLOR_YCrCb2BGR);
	return src_2;
}
*/

vector<vector<Point>> find_contours(Mat src)
{
	vector<vector<Point>> contours;

	int height = src.rows, width = src.cols;
	Mat dst(height, width, CV_8UC1, Scalar(0));

	int i, j, h;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			if ((int)src.at<uchar>(i, j) == 255 && (int)dst.at<uchar>(i, j) == 0)
			{
				vector<Point> current_contour;
				Point P0;
				P0.y = i;
				P0.x = j;

				Point Pn = P0;

				int dir = 7;
				bool open_contour = false;
				do
				{
					int curr_i = Pn.y;
					int curr_j = Pn.x;
					dst.at<uchar>(curr_i, curr_j) = 255;
					current_contour.push_back(Point(curr_j, curr_i));

					if (dir % 2 == 0)
					{
						dir += 7;
					}
					else
					{
						dir += 6;
					}
					dir %= 8;

					h = dir;
					int counter = 0;
					while (counter < 8)
					{
						int next_i = curr_i + di[h];
						int next_j = curr_j + dj[h];
						if (is_inside(src, next_i, next_j) && (int)src.at<uchar>(next_i, next_j) == 255 && (int)dst.at<uchar>(next_i, next_j) == 0)
						{
							Pn.y = next_i;
							Pn.x = next_j;
							dir = h;

							break;
						}
						h = (h + 1) % 8;
						++counter;
					}

					if (counter == 8)
					{
						open_contour = true;
						break;
					}
				}while (P0 != Pn);

				if (current_contour.size() > 10)
				{
					if (!open_contour)
					{
						// remove the last two elements which are equal to the first two elements
						current_contour.pop_back();
					}
					contours.push_back(current_contour);
				}
			}
		}
	}

	return contours;
}

double point_to_line_dist(Point point, Point line_start, Point line_end)
{
	double x = point.x;
	double y = point.y;
	double x1 = line_start.x;
	double y1 = line_start.y;
	double x2 = line_end.x;
	double y2 = line_end.y;

	return abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / sqrt(pow(y2 - y1, 2) + pow(x2 - x1, 2));
}

vector<Point> approx_poly_dp(vector<Point> contour, double epsilon) 
{
	int n = contour.size();
	if (n < 3)
	{
		return contour;
	}

	double max_dist = 0;
	int index = 0, end = n - 1;
	int i;
	for (i = 1; i < end; ++i)
	{
		double dist = point_to_line_dist(contour[i], contour[0], contour[end]);
		if (dist > max_dist)
		{
			max_dist = dist;
			index = i;
		}
	}

	vector<Point> approx_contour;
	if (max_dist > epsilon) 
	{
		vector<Point> first_part, second_part;

		for (i = 0; i <= index; ++i)
		{
			first_part.push_back(contour[i]);
		}
		for (i = index; i < n; ++i)
		{
			second_part.push_back(contour[i]);
		}

		vector<Point> first_part_approx = approx_poly_dp(first_part, epsilon);
		vector<Point> second_part_approx = approx_poly_dp(second_part, epsilon);

		approx_contour.insert(approx_contour.end(), first_part_approx.begin(), first_part_approx.end() - 1);
		approx_contour.insert(approx_contour.end(), second_part_approx.begin(), second_part_approx.end());
	}
	else 
	{
		approx_contour.push_back(contour[0]);
		approx_contour.push_back(contour[end]);
	}

	return approx_contour;
}

void draw_contours(Mat& image, vector<vector<Point>> contours)
{
	int n = contours.size(), i, j;
	for (i = 0; i < n; ++i)
	{
		vector<Point> contour = contours[i];
		int m = contours[i].size();
		for (j = 0; j < m; ++j)
		{
			line(image, contour[j], contour[(j + 1) % m], Scalar(255), 1);
		}
	}

	return;
}

int calculate_otsu_threshold(Mat src) 
{
	int height = src.rows, width = src.cols;
	int hist[256] = { 0 };
	int number_of_pixels = height * width;

	histogram(src, hist);

	double pdf[256] = { 0 };
	int i;
	for (i = 0; i < 256; ++i)
	{
		pdf[i] = (double)hist[i] / (double)number_of_pixels;
	}

	double cdf[256] = { 0 };
	cdf[0] = pdf[0];
	for (i = 1; i < 256; ++i)
	{
		cdf[i] = cdf[i - 1] + pdf[i];
	}

	double cdf_mean[256] = { 0 };
	cdf_mean[0] = 0;
	for (i = 1; i < 256; ++i)
	{
		cdf_mean[i] = cdf_mean[i - 1] + i * pdf[i];
	}

	double max_variance = 0;
	int threshold = 0;
	for (i = 0; i < 256; ++i) 
	{
		double p_background = cdf[i];
		double p_foreground = 1 - p_background;
		double m_background = cdf_mean[i] / p_background;
		double m_foreground = (cdf_mean[255] - cdf_mean[i]) / p_foreground;
		double between_class_variance = p_background * p_foreground * (m_background - m_foreground) * (m_background - m_foreground);
		if (between_class_variance > max_variance)
		{
			max_variance = between_class_variance;
			threshold = i;
		}
	}

	return threshold;
}

Mat threshold_image(Mat image, int threshold) 
{
	int height = image.rows, width = image.cols;
	Mat binary_image(height, width, CV_8UC1, Scalar(0));
	int i, j;
	for (i = 0; i < height; ++i) 
	{
		for (j = 0; j < width; ++j) 
		{
			if ((int)image.at<uchar>(i, j) > threshold) 
			{
				binary_image.at<uchar>(i, j) = 255;
			}
		}
	}

	return binary_image;
}

struct Region
{
	vector<Point> points;
	int size;
};

void delete_regions(vector<Region>& regions, vector<Region> regions_to_be_removed)
{
	int n = regions.size(), i, j;
	vector<Region>::iterator it;
	for (it = regions_to_be_removed.begin(); it != regions_to_be_removed.end(); ++it)
	{
		Region current_region_to_be_removed = (*it);
		for (i = n - 1; i >= 0; --i)
		{
			if (regions[i].points == current_region_to_be_removed.points && regions[i].size == current_region_to_be_removed.size)
			{
				for (j = i; j < n - 1; ++j)
				{
					regions[j] = regions[j + 1];
				}
				--n;
			}
		}
	}
	regions.resize(n);

	return;
}

void delete_redundant_regions(vector<Region>& regions, int src_height, int src_width)
{
	vector<Region> regions_to_be_removed;
	int n = regions.size();
	int i, j;

	// first remove regions for which the bounding box touches at least one of the edges of the image
	for (i = 0; i < n; ++i)
	{
		Rect rectangle = boundingRect(regions[i].points);
		if (rectangle.x == 0 || rectangle.x + rectangle.width == src_width ||
			rectangle.y == 0 || rectangle.y + rectangle.height == src_height)
		{
			regions_to_be_removed.push_back(regions[i]);
		}
	}
	delete_regions(regions, regions_to_be_removed);
	regions_to_be_removed.clear();

	n = regions.size();
	// then, remove small regions inside larger regions
	for (i = 0; i < n - 1; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{
			Rect rectangle_i = boundingRect(regions[i].points);
			Rect rectangle_j = boundingRect(regions[j].points);
			int rectangle_i_size = rectangle_i.width * rectangle_i.height;
			int rectangle_j_size = rectangle_j.width * rectangle_j.height;
			if (rectangle_j_size < rectangle_i_size &&
				is_inside_square(rectangle_j.x, rectangle_j.x + rectangle_j.width, rectangle_j.y, rectangle_j.y + rectangle_j.height, rectangle_i.x, rectangle_i.x + rectangle_i.width, rectangle_i.y, rectangle_i.y + rectangle_i.height))
			{
				regions_to_be_removed.push_back(regions[j]);
			}
			else if (rectangle_i_size < rectangle_j_size &&
					 is_inside_square(rectangle_i.x, rectangle_i.x + rectangle_i.width, rectangle_i.y, rectangle_i.y + rectangle_i.height, rectangle_j.x, rectangle_j.x + rectangle_j.width, rectangle_j.y, rectangle_j.y + rectangle_j.height))
			{
				regions_to_be_removed.push_back(regions[i]);
			}
		}
	}
	delete_regions(regions, regions_to_be_removed);

	return;
}

vector<Region> my_MSER(Mat src, int min_area, int max_area, int width_height_threshold)
{
	int height = src.rows, width = src.cols;
	vector<Region> regions;
	Mat visited(width, height, CV_8UC1, Scalar(0));
	vector<Point> stack;

	int i, j;
	for (i = 0; i < height; ++i)
	{
		for (j = 0; j < width; ++j)
		{
			if ((int)visited.at<uchar>(j, i) == 0) 
			{
				Point point = Point(j, i);

				int intensity = (int)src.at<uchar>(i, j);
				stack.push_back(point);
				visited.at<uchar>(j, i) = 1;

				Region region;
				region.points.push_back(point);
				region.size = 1;

				while (!stack.empty()) 
				{
					Point current_point = stack.back();
					stack.pop_back();

					int k;
					for (k = 0; k < 8; ++k) 
					{
						int new_j = current_point.x + dj[k];
						int new_i = current_point.y + di[k];

						if (is_inside(src, new_i, new_j) && (int)visited.at<uchar>(new_j, new_i) == 0 && (int)src.at<uchar>(new_i, new_j) == intensity)
						{
							Point new_point = Point(new_j, new_i);

							stack.push_back(new_point);
							visited.at<uchar>(new_j, new_i) = 1;
							region.points.push_back(new_point);
							++region.size;
						}
					}
				}

				if (region.size >= min_area && region.size <= max_area)
				{
					Rect rectangle = boundingRect(region.points);
					if (abs(rectangle.width - rectangle.height) <= width_height_threshold)
					{
						regions.push_back(region);
					}
				}
			}
		}
	}

	delete_redundant_regions(regions, height, width);

	return regions;
}

bool is_inside_outline(int x_min_1, int x_max_1, int y_min_1, int y_max_1, int x_min_2, int x_max_2, int y_min_2, int y_max_2)
{
	int length_1 = x_max_1 - x_min_1;
	int center_x_1 = x_min_1 + length_1 / 2;
	int center_y_1 = y_min_1 + length_1 / 2;

	return ((x_min_2 <= x_min_1 && x_min_1 <= x_max_2 && y_min_2 <= y_min_1 && y_min_1 <= y_max_2) &&
			(x_min_2 <= x_min_1 && x_min_1 <= x_max_2 && y_min_2 <= y_max_1 && y_max_1 <= y_max_2) &&
			(x_min_2 <= x_max_1 && x_max_1 <= x_max_2 && y_min_2 <= y_min_1 && y_min_1 <= y_max_2) &&
			(x_min_2 <= x_max_1 && x_max_1 <= x_max_2 && y_min_2 <= y_max_1 && y_max_1 <= y_max_2)) ||
		   (x_min_2 <= center_x_1 && center_x_1 <= x_max_2 && y_min_2 <= center_y_1 && center_y_1 <= y_max_2);
}

void filter_bounding_boxes(vector<Rect>& possible_traffic_sign_outlines)
{
	int n = possible_traffic_sign_outlines.size(), i, j;
	if (n == 0)
	{
		return;
	}

	int average_area = 0;

	for (i = 0; i < n; ++i)
	{
		int area = possible_traffic_sign_outlines[i].width * possible_traffic_sign_outlines[i].height;
		average_area += area;
	}
	average_area /= n;

	vector<Rect> outlines_to_be_removed;
	for (i = 0; i < n - 1; ++i)
	{
		for (j = i + 1; j < n; ++j)
		{
			int area_i = possible_traffic_sign_outlines[i].height * possible_traffic_sign_outlines[i].width;
			int x_min_i = possible_traffic_sign_outlines[i].x;
			int x_max_i = x_min_i + possible_traffic_sign_outlines[i].width;
			int y_min_i = possible_traffic_sign_outlines[i].y;
			int y_max_i = y_min_i + possible_traffic_sign_outlines[i].height;

			int area_j = possible_traffic_sign_outlines[j].height * possible_traffic_sign_outlines[j].width;
			int x_min_j = possible_traffic_sign_outlines[j].x;
			int x_max_j = x_min_j + possible_traffic_sign_outlines[j].width;
			int y_min_j = possible_traffic_sign_outlines[j].y;
			int y_max_j = y_min_j + possible_traffic_sign_outlines[j].height;

			if (area_i < area_j && is_inside_outline(x_min_i, x_max_i, y_min_i, y_max_i, x_min_j, x_max_j, y_min_j, y_max_j))
			{
				outlines_to_be_removed.push_back(possible_traffic_sign_outlines[i]);
			}
			else if (area_j < area_i && is_inside_outline(x_min_j, x_max_j, y_min_j, y_max_j, x_min_i, x_max_i, y_min_i, y_max_i))
			{
				outlines_to_be_removed.push_back(possible_traffic_sign_outlines[j]);
			}
		}
	}

	vector<Rect>::iterator it;
	for (it = outlines_to_be_removed.begin(); it != outlines_to_be_removed.end(); ++it)
	{
		Rect current_outline_to_be_removed = (*it);
		for (i = n - 1; i >= 0; --i)
		{
			if (possible_traffic_sign_outlines[i] == current_outline_to_be_removed)
			{
				for (j = i; j < n - 1; ++j)
				{
					possible_traffic_sign_outlines[j] = possible_traffic_sign_outlines[j + 1];
				}
				--n;
			}
		}
	}
	possible_traffic_sign_outlines.resize(n);

	return;
}

bool createDirectory(string path) 
{
	if (CreateDirectoryA(path.c_str(), NULL) || GetLastError() == ERROR_ALREADY_EXISTS) 
	{
		return true;
	}
	else 
	{
		cerr << "Failed to create directory: " << path << '\n';
		return false;
	}
}

void generate_all_traffic_sign_images()
{
	string folder_path = "C:\\Users\\Tudor Cristea\\Documents\\IP\\project\\datasets\\archive\\MyTest";
	string search_path = folder_path + "\\*.*";

	int i, j, n;
	WIN32_FIND_DATAA file_data;
	HANDLE hFind = FindFirstFileA(search_path.c_str(), &file_data);
	if (hFind != INVALID_HANDLE_VALUE)
	{
		do
		{
			if (!(file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
			{
				string filename = file_data.cFileName;
				string full_path = folder_path + "\\" + filename;
				Mat src_color = imread(full_path, IMREAD_COLOR);
				if (!src_color.empty())
				{
					cout << filename << "\n";
					int height = src_color.rows, width = src_color.cols;
					Mat src_gray = rgb_2_gray(src_color);

					int otsu_threshold = calculate_otsu_threshold(src_gray);
					Mat src_binary = threshold_image(src_gray, otsu_threshold);

					vector<Rect> possible_traffic_sign_outlines;

					vector<Region> regions = my_MSER(src_binary, 450, height * width / 5, 30);

					Mat src_mser = src_color.clone();
					n = regions.size(), i;
					for (i = 0; i < n; ++i)
					{
						Rect bounding_box = boundingRect(regions[i].points);
						possible_traffic_sign_outlines.push_back(bounding_box);
					}

					Mat src_hsv = rgb_2_hsv(src_color);
					vector<Mat> src_filtered_pair = filter_colors(src_color, src_hsv);
					Mat src_filtered = src_filtered_pair[0];
					Mat src_filtered_red = src_filtered_pair[1];

					Mat src_canny_red = canny(src_filtered_red);
					Mat src_canny = canny(src_filtered);

					vector<vector<Point>> contours = find_contours(src_canny_red);
					Mat src_contours(height, width, CV_8UC1, Scalar(0));
					for (i = 0; i < contours.size(); ++i)
					{
						for (j = 0; j < contours[i].size(); ++j)
						{
							src_contours.at<uchar>(contours[i][j].y, contours[i][j].x) = 255;
						}
					}

					float percentage = 0.1;
					Mat src_approx_contours(height, width, CV_8UC1, Scalar(0));
					n = contours.size();
					vector<vector<Point>> approx_contours;
					for (i = 0; i < n; ++i)
					{
						vector<Point> approx_contour = approx_poly_dp(contours[i], contours[i].size() * PI / 4 * percentage);
						approx_contours.push_back(approx_contour);
					}
					draw_contours(src_approx_contours, approx_contours);

					Mat src_approx_contours_points(height, width, CV_8UC1, Scalar(0));
					for (i = 0; i < approx_contours.size(); ++i)
					{
						for (j = 0; j < approx_contours[i].size(); ++j)
						{
							src_approx_contours_points.at<uchar>(approx_contours[i][j].y, approx_contours[i][j].x) = 255;
						}
					}


					Mat src_hough_circle = src_color.clone();
					vector<Vec3i> circles = hough_circle(src_canny);
					n = circles.size();
					for (i = 0; i < n; ++i)
					{
						int x = circles[i][1], y = circles[i][0], radius = circles[i][2];
						Point center = Point(x, y);

						if (is_inside(src_hough_circle, center.y - radius, center.x - radius) && is_inside(src_hough_circle, center.y + radius, center.x + radius))
						{
							Rect rect(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
							possible_traffic_sign_outlines.push_back(rect);
						}
					}


					Mat src_hough_square = src_color.clone();
					vector<Vec3i> squares = hough_square(src_canny);
					n = squares.size();
					for (i = 0; i < n; ++i)
					{
						int x = squares[i][1], y = squares[i][0], length = squares[i][2];
						Point center = { x, y };

						if (is_inside(src_hough_square, center.y - length, center.x - length) && is_inside(src_hough_square, center.y + length, center.x + length))
						{
							Rect rect(center.x - length, center.y - length, 2 * length, 2 * length);
							possible_traffic_sign_outlines.push_back(rect);
						}
					}


					Mat src_harris_triangle = src_color.clone();
					vector<Triangle> triangles = harris_corner_detection_triangles(src_approx_contours, src_approx_contours_points);
					n = triangles.size();
					for (i = 0; i < n; ++i)
					{
						Point A = triangles[i].A;
						Point B = triangles[i].B;
						Point C = triangles[i].C;
						Point O = (A + B + C) / 3;
						int length = (int)ceil(max(max(dist(A, B), dist(A, C)), dist(B, C)));

						if (is_inside(src_harris_triangle, O.y - length / 2, O.x - length / 2) && is_inside(src_harris_triangle, O.y + length / 2, O.x + length / 2))
						{
							Rect rect(O.x - length / 2, O.y - length / 2, length, length);
							possible_traffic_sign_outlines.push_back(rect);
						}
					}

					filter_bounding_boxes(possible_traffic_sign_outlines);

					string image_dir = folder_path + "\\result_images\\" + filename.substr(0, filename.find_last_of('.'));
					if (createDirectory(image_dir))
					{
						// creating separate images from the detected shapes
						n = possible_traffic_sign_outlines.size();
						for (i = 0; i < n; ++i)
						{
							int current_height = possible_traffic_sign_outlines[i].height;
							int current_width = possible_traffic_sign_outlines[i].width;
							int new_x = possible_traffic_sign_outlines[i].x - min(5, current_width / 8);
							int new_y = possible_traffic_sign_outlines[i].y - min(5, current_height / 8);
							int new_height = current_height + min(10, current_height / 4);
							int new_width = current_width + min(10, current_width / 4);

							if (is_inside(src_color, new_y, new_x) && is_inside(src_color, new_y + new_height, new_x + new_width))
							{
								Rect extended_outline(new_x, new_y, new_width, new_height);
								string name = image_dir + "\\" + filename.substr(0, filename.find_last_of('.')) + "_" + to_string(i) + ".png";
								imwrite(name, src_color(extended_outline));
							}
						}
					}
				}
			}
		} while (FindNextFileA(hFind, &file_data) != 0);
		FindClose(hFind);
	}
	else
	{
		cerr << "Could not open directory\n";
		return;
	}

	return;
}

int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	//generate_all_traffic_sign_images();

	int i, j, n;
	Mat src_color;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src_color = imread(fname, IMREAD_COLOR);
		//imwrite("example/original.png", src_color);
		int height = src_color.rows, width = src_color.cols;
		Mat src_gray = rgb_2_gray(src_color);
		
		int otsu_threshold = calculate_otsu_threshold(src_gray);
		Mat src_binary = threshold_image(src_gray, otsu_threshold);
		//imshow("otsu", src_binary);
		//imwrite("example/otsu.png", src_binary);

		vector<Rect> possible_traffic_sign_outlines;

		vector<Region> regions = my_MSER(src_binary, 450, height * width / 5, 30);

		Mat src_mser = src_color.clone();
		n = regions.size(), i;
		for (i = 0; i < n; ++i) 
		{
			Rect bounding_box = boundingRect(regions[i].points);
			rectangle(src_mser, bounding_box, Scalar(0, 0, 255), 1, LINE_AA);
			possible_traffic_sign_outlines.push_back(bounding_box);
		}
		imshow("MSER", src_mser);
		//imwrite("example/mser.png", src_mser);
		
		
		////Mat src_color_eq = equalize_histogram(src_color);
		////imwrite("example/equalized.png", src_color_eq);
		Mat src_hsv = rgb_2_hsv(src_color);
		vector<Mat> src_filtered_pair = filter_colors(src_color, src_hsv);
		Mat src_filtered = src_filtered_pair[0];
		Mat src_filtered_red = src_filtered_pair[1];
		//imshow("filtered", src_filtered);
		//imwrite("example/filtered.png", src_filtered);

		////Mat src_canny_full = canny(src_color);
		Mat src_canny_red = canny(src_filtered_red);
		Mat src_canny = canny(src_filtered);
		imshow("canny", src_canny);
		//imwrite("example/canny.png", src_canny);
		
		vector<vector<Point>> contours = find_contours(src_canny_red);
		Mat src_contours(height, width, CV_8UC1, Scalar(0));
		for (i = 0; i < contours.size(); ++i)
		{
			for (j = 0; j < contours[i].size(); ++j)
			{
				src_contours.at<uchar>(contours[i][j].y, contours[i][j].x) = 255;
			}
		}
		////imshow("contours", src_contours);

		float percentage = 0.1;
		Mat src_approx_contours(height, width, CV_8UC1, Scalar(0));
		n = contours.size();
		vector<vector<Point>> approx_contours;
		for (i = 0; i < n; ++i)
		{
			vector<Point> approx_contour = approx_poly_dp(contours[i], contours[i].size() * PI / 4 * percentage);
			approx_contours.push_back(approx_contour);
		}
		draw_contours(src_approx_contours, approx_contours);
		//imshow("approx_poly_dp", src_approx_contours);
		//imwrite("example/approx_poly_dp.png", src_approx_contours);
		Mat src_approx_contours_points(height, width, CV_8UC1, Scalar(0));
		for (i = 0; i < approx_contours.size(); ++i)
		{
			for (j = 0; j < approx_contours[i].size(); ++j)
			{
				src_approx_contours_points.at<uchar>(approx_contours[i][j].y, approx_contours[i][j].x) = 255;
			}
		}
		//imshow("approx_poly_dp_points", src_approx_contours_points);
		//imwrite("example/approx_poly_dp_points.png", src_approx_contours_points);

		
		Mat src_hough_circle = src_color.clone();
		vector<Vec3i> circles = hough_circle(src_canny);
		n = circles.size();
		for (i = 0; i < n; ++i)
		{
			int x = circles[i][1], y = circles[i][0], radius = circles[i][2];
			Point center = Point(x, y);

			if (is_inside(src_hough_circle, center.y - radius, center.x - radius) && is_inside(src_hough_circle, center.y + radius, center.x + radius))
			{
				Rect rect(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
				possible_traffic_sign_outlines.push_back(rect);

				circle(src_hough_circle, center, 1, Scalar(255, 0, 0), 2, LINE_AA);
				circle(src_hough_circle, center, radius, Scalar(0, 255, 0), 2, LINE_AA);
			}
		}
		imshow("hough circle", src_hough_circle);
		//imwrite("example/circles.png", src_hough_circle);

		
		Mat src_hough_square = src_color.clone();
		vector<Vec3i> squares = hough_square(src_canny);
		n = squares.size();
		for (i = 0; i < n; ++i)
		{
			int x = squares[i][1], y = squares[i][0], length = squares[i][2];
			Point center = { x, y };
			Point A = { x - length, y - length };
			Point B = { x - length, y + length };
			Point C = { x + length, y + length };
			Point D = { x + length, y - length };

			if (is_inside(src_hough_square, center.y - length, center.x - length) && is_inside(src_hough_square, center.y + length, center.x + length))
			{
				Rect rect(center.x - length, center.y - length, 2 * length, 2 * length);
				possible_traffic_sign_outlines.push_back(rect);

				circle(src_hough_square, center, 1, Scalar(0, 0, 255), 2, LINE_AA);
				line(src_hough_square, A, B, Scalar(0, 255, 255), 2, LINE_AA);
				line(src_hough_square, B, C, Scalar(0, 255, 255), 2, LINE_AA);
				line(src_hough_square, C, D, Scalar(0, 255, 255), 2, LINE_AA);
				line(src_hough_square, D, A, Scalar(0, 255, 255), 2, LINE_AA);
			}
		}
		imshow("hough square", src_hough_square);
		//imwrite("example/squares.png", src_hough_square);
		
		
		Mat src_harris_triangle = src_color.clone();
		vector<Triangle> triangles = harris_corner_detection_triangles(src_approx_contours, src_approx_contours_points);
		n = triangles.size();
		for (i = 0; i < n; ++i)
		{
			Point A = triangles[i].A;
			Point B = triangles[i].B;
			Point C = triangles[i].C;
			Point O = (A + B + C) / 3;
			int length = (int)ceil(max(max(dist(A, B), dist(A, C)), dist(B, C)));
			
			if (is_inside(src_harris_triangle, O.y - length / 2, O.x - length / 2) && is_inside(src_harris_triangle, O.y + length / 2, O.x + length / 2))
			{
				Rect rect(O.x - length / 2, O.y - length / 2, length, length);
				possible_traffic_sign_outlines.push_back(rect);

				circle(src_harris_triangle, (A + B + C) / 3, 1, Scalar(203, 192, 255), 2, LINE_AA);
				line(src_harris_triangle, A, B, Scalar(128, 0, 128), 2, LINE_AA);
				line(src_harris_triangle, B, C, Scalar(128, 0, 128), 2, LINE_AA);
				line(src_harris_triangle, C, A, Scalar(128, 0, 128), 2, LINE_AA);
			}
		}
		imshow("harris triangles", src_harris_triangle);
		//imwrite("example/triangles.png", src_harris_triangle);
		
		
		Mat src_color_semifinal = src_color.clone();
		n = possible_traffic_sign_outlines.size();
		for (i = 0; i < n; ++i)
		{
			rectangle(src_color_semifinal, possible_traffic_sign_outlines[i], Scalar(0, 0, 255), 1, LINE_AA);
		}
		imshow("semifinal", src_color_semifinal);
		//imwrite("example/semifinal.png", src_color_semifinal);

		filter_bounding_boxes(possible_traffic_sign_outlines);

		// creating separate images from the detected shapes
		Mat src_color_final = src_color.clone();
		n = possible_traffic_sign_outlines.size();
		for (i = 0; i < n; ++i)
		{
			int current_height = possible_traffic_sign_outlines[i].height;
			int current_width = possible_traffic_sign_outlines[i].width;
			int new_x = possible_traffic_sign_outlines[i].x - min(5, current_width / 8);
			int new_y = possible_traffic_sign_outlines[i].y - min(5, current_height / 8);
			int new_height = current_height + min(10, current_height / 4);
			int new_width = current_width + min(10, current_width / 4);

			if (is_inside(src_color, new_y, new_x) && is_inside(src_color, new_y + new_height, new_x + new_width))
			{
				Rect extended_outline(new_x, new_y, new_width, new_height);
				rectangle(src_color_final, extended_outline, Scalar(0, 0, 255), 1, LINE_AA);
				//imshow(to_string(i), src_color(extended_outline));
				//imwrite(to_string(i), src_color(extended_outline));
			}
		}

		imshow("final", src_color_final);
		//imwrite("example/final.png", src_color_final);
		waitKey(0);
	}

	return 0;
}