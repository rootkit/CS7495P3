/**
 * \file TestKAZE.cpp
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


#include <cppunit/TestFixture.h>
#include <cppunit/extensions/HelperMacros.h>

#include <KAZE.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class TestKaze : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE( TestKaze );
	CPPUNIT_TEST(TestFeatureExtraction);
	CPPUNIT_TEST_SUITE_END();
public:

	virtual void setUp()
	{
		srand(10);
	}

	virtual void tearDown () {}

	void defaultOptions(toptions& options)
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

	void TestFeatureExtraction()
	{
		toptions options;
		cv::Mat img, img_32, img_rgb;

		defaultOptions(options);

		img = cv::imread("/home/arprice/Desktop/platypus.jpg", 0);

		CPPUNIT_ASSERT(img.data != NULL);

		// Convert the image to float
		img.convertTo(img_32,CV_32F,1.0/255.0,0);
		img_rgb = cv::Mat(cv::Size(img.cols,img.rows),CV_8UC3);

		options.img_width = img.cols;
		options.img_height = img.rows;

		// Create the KAZE object
		KAZE evolution(options);

		// Create the nonlinear scale space
		evolution.Create_Nonlinear_Scale_Space(img_32);

		std::vector<cv::KeyPoint> kpts;
		cv::Mat desc;

		evolution.Feature_Detection(kpts);
		evolution.Feature_Description(kpts,desc);

		if (options.show_results)
		{
			std::cout << "Time Scale Space: " << evolution.Get_Time_NLScale() << std::endl;
			std::cout << "Time Detector: " << evolution.Get_Time_Detector() << std::endl;
			std::cout << "Time Descriptor: " << evolution.Get_Time_Descriptor() << std::endl;
			std::cout << "Number of Keypoints: " << kpts.size() << std::endl;

			// Create the OpenCV window
			cv::namedWindow("Image",CV_WINDOW_FREERATIO);

			// Copy the input image to the color one
			cv::cvtColor(img,img_rgb,CV_GRAY2BGR);

			// Draw the list of detected points
			draw_keypoints(img_rgb,kpts);

			cv::imshow("Image",img_rgb);
			cv::waitKey(0);

			// Destroy the windows
			cv::destroyAllWindows();

		}

	}


};

CPPUNIT_TEST_SUITE_REGISTRATION(TestKaze);
