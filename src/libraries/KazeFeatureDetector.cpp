/**
 * \file KazeFeatureDetector.cpp
 * \brief
 *
 * \author Andrew Price
 * \date 11 22, 2013
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

#include "KazeFeatureDetector.h"

namespace cv
{

static void defaultOptions(toptions& options)
{
	options.soffset = DEFAULT_SCALE_OFFSET;
	options.omax = DEFAULT_OCTAVE_MAX;
	options.nsublevels = DEFAULT_NSUBLEVELS;
	options.dthreshold = DEFAULT_DETECTOR_THRESHOLD;
	options.diffusivity = DEFAULT_DIFFUSIVITY_TYPE;
	options.descriptor = DEFAULT_DESCRIPTOR_MODE;
	options.upright = DEFAULT_UPRIGHT;
	options.extended = DEFAULT_EXTENDED;
	options.sderivatives = DEFAULT_SIGMA_SMOOTHING_DERIVATIVES;
	options.save_scale_space = DEFAULT_SAVE_SCALE_SPACE;
	options.show_results = DEFAULT_SHOW_RESULTS;
	options.save_keypoints = DEFAULT_SAVE_KEYPOINTS;
}

KazeFeatureDetector::KazeFeatureDetector()
{
	usePreloadedEnvironment = false;
}

void KazeFeatureDetector::preloadEnvironment(KazePtr& e)
{
	this->environment = e;
	usePreloadedEnvironment = true;
}

void KazeFeatureDetector::detectImpl( const Mat& image, vector<KeyPoint>& keypoints, const Mat& mask ) const
{
	if (usePreloadedEnvironment)
	{
		environment->Feature_Detection(keypoints);
	}
	else
	{
		toptions options;
		defaultOptions(options);
		options.img_width = image.cols;
		options.img_height = image.rows;

		std::cerr << "Creating KAZE Evolution." << std::endl;
		KAZE evolution(options);

		cv::Mat img_32;
		cv::Mat bw;
		if (image.channels() == 1)
		{
			image.convertTo(img_32,CV_32F,1.0/255.0,0);
		}
		else
		{
			cv::cvtColor(image, bw, CV_RGB2GRAY);
			bw.convertTo(img_32,CV_32F,1.0/255.0,0);
		}

		std::cerr << "Creating NLSS." << std::endl;
		evolution.Create_Nonlinear_Scale_Space(img_32);
		std::cerr << "Detecting Features." << std::endl;
		evolution.Feature_Detection(keypoints);
	}
}

}
