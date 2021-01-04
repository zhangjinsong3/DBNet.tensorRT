//
// Created by zjs on 19-3-27.
//
// This head saves some python-like functions for better coding on C++ !
#pragma once

#include <algorithm>
#include <stdlib.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

template<class ForwardIterator>
inline size_t argmin(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::min_element(first, last));
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last)
{
    return std::distance(first, std::max_element(first, last));
}

template <typename T>
inline bool in(int value, std::vector<T>& items)
{
    for (auto& item : items)
        if (item == value)
            return true;
    return false;
}

template <typename T>
inline T Sum(std::vector<T>& vec)
{
    T res = 0;
    for (size_t i=0; i<vec.size(); i++)
    {
        res += vec[i];
    }
    return res;
}