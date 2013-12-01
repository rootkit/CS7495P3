/**
 * \file TestBOW.cpp
 * \brief
 *
 * \author Andrew Price
 * \date 11 23, 2013
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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>

class TestBoW : public CppUnit::TestFixture
{
	CPPUNIT_TEST_SUITE( TestBoW );
	CPPUNIT_TEST(TestCluster);
	CPPUNIT_TEST_SUITE_END();
public:

	virtual void setUp()
	{
		srand(10);
		cv::initModule_nonfree();
		cv::initModule_features2d();
	}

	virtual void tearDown () {}

	cv::Mat trainVocabulary(const std::vector<cv::Mat>& templates,
							const cv::Ptr<cv::FeatureDetector> detector,
							const cv::Ptr<cv::DescriptorExtractor> extractor,
							cv::Ptr<cv::BOWTrainer> trainer)
	{
		// Use templates to train a vocabulary
		cv::Mat desc;
		for(cv::Mat temp : templates)
		{
			std::vector<cv::KeyPoint> kp;
			detector->detect(temp, kp);
			extractor->compute(temp, kp, desc);
			trainer->add(desc);
		}

		return trainer->cluster();
	}

	void TestCluster()
	{
		cv::Ptr<cv::FeatureDetector> detector(new cv::SiftFeatureDetector());
		cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SiftDescriptorExtractor());
		cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher());

		cv::Ptr<cv::BOWTrainer> trainer(new cv::BOWKMeansTrainer(5));
		cv::Ptr<cv::BOWImgDescriptorExtractor> imgExtractor(new cv::BOWImgDescriptorExtractor(extractor, matcher));


	}
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBoW);
