/**
 * \file voc_utils.cpp
 * \brief
 *
 * \author Andrew Price
 * \date 12 2, 2013
 *
 * \copyright
 *
 * Copyright (c) 2013, Georgia Tech Research Corporation
 * All rights reserved.
 *
 * Humanoid Robotics Lab Georgia Institute of Technology
 * Director: Mike Stilman http://www.golems.org
 *
 * This file is provided under the following "BSD-style" License:
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above
 *   copyright notice, this list of conditions and the following
 *   disclaimer in the documentation and/or other materials provided
 *   with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
 * USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "voc_utils.h"

// For file reader
#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

const VOCFileStructure VOC_FILE_STRUCTURE;

// Parses a VOC file to get member filenames
void getMemberList(const std::string& filename, std::vector<std::string>& files, int numToRead)
{
	std::ifstream readFile(filename);
	int linesRead = 0;

	// Read file
	while(readFile && (numToRead < 0 || linesRead < numToRead))
	{
		// Read row
		std::string	line;
		std::getline(readFile, line);

		std::stringstream lineStream(line);
		std::string	target, val;

		std::getline(lineStream, target, ' ');
		std::getline(lineStream, val);
		if (!(val.find("-1") < val.length()))
		{
			files.push_back(target);
			++linesRead;
		}
	}
}

void readVOCLists(const std::string directory, TrainingSet& collections, int numToRead)
{
	int numRead = 0;
	boost::filesystem3::directory_iterator iter(directory);
	for (boost::filesystem3::directory_iterator end; iter != end; ++iter)
	{
		std::string filename = iter->path().parent_path().string() + std::string("/")
				+ (iter->path()).filename().string();
		if (filename.find("_train.txt") < filename.length())
		{
			std::vector<std::string> members;
			std::stringstream lineStream(iter->path().filename().string());
			std::string	category;

			std::getline(lineStream, category, '_');
			getMemberList(filename, members, 20);

			collections.insert(std::pair<std::string, std::vector<std::string> >(category, members));
			++numRead;
		}
		if (numToRead > 0 && numRead >= numToRead) {break;}
	}
}

cv::Mat getMask(const std::string filename, int dilationRadius)
{
	cv::Mat readImg = cv::imread(VOC_FILE_STRUCTURE.maskDir() + filename + ".png");

	if (readImg.data == NULL)
	{
		return cv::Mat();
	}

	assert(readImg.type() == CV_8UC3);

	cv::cvtColor(readImg, readImg, CV_RGB2GRAY);

	cv::Mat mask;
	cv::threshold(readImg, mask, 1, 255, CV_THRESH_BINARY);

	cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
										 cv::Size( 2*dilationRadius + 1, 2*dilationRadius+1 ),
										 cv::Point( dilationRadius, dilationRadius ) );
	cv::dilate(mask, mask, element);

	//cv::imshow("mask", mask);
	//cv::waitKey();

	return mask;
}

void validateKeypoints(std::vector<cv::KeyPoint>& kps, const cv::Mat& mask)
{
	if (mask.data == NULL)
	{
		// No mask found, so no valid keypoints
		kps = std::vector<cv::KeyPoint>();
		return;
	}

	for (std::vector<cv::KeyPoint>::iterator i = kps.begin(); i != kps.end(); ++i)
	{
		cv::KeyPoint kp = *i;
		if (mask.at<uchar>(kp.pt) == 0)
		{
			kps.erase(i);
			--i;
		}
	}
}
