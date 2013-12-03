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
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ml/ml.hpp>

#include <boost/filesystem.hpp>

const int BOW_FEATURE_SIZE = 16;
const std::string BASE_PATH = "/home/arprice/Desktop/bow/";
const std::string TEMPLATES_PATH = "templates/";
const std::string TRAINING_PATH = "training/";
const std::string TESTING_PATH = "testing/";

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

	cv::Mat buildVocabulary(boost::filesystem3::directory_iterator& iter,
							const cv::Ptr<cv::FeatureDetector> detector,
							const cv::Ptr<cv::DescriptorExtractor> extractor,
							cv::Ptr<cv::BOWTrainer> trainer,
							const bool lookForSavedVocab = true,
							const bool saveTrainedVocab = true)
	{
		cv::Mat vocabulary;
		std::string parentPath = iter->path().parent_path().string();
		std::string vocabPath = parentPath + "/vocab.xml";

		// Look for pretrained vocabulary
		if (lookForSavedVocab && boost::filesystem3::exists(vocabPath))
		{
			std::cout << "Loading filesystem vocabulary: " << vocabPath << std::endl;
			cv::FileStorage fs(iter->path().parent_path().string() + "/vocab.xml", cv::FileStorage::READ);
			fs["vocabulary"] >> vocabulary;
			fs.release();

			return vocabulary;
		}

		// Use templates to train a vocabulary
		cv::Mat desc;
		for (boost::filesystem3::directory_iterator end; iter != end; ++iter)
		{
			if (iter->path().filename().string() == "vocab.xml") {continue;}
			std::string filename = parentPath + "/" + iter->path().filename().string();
			cv::Mat temp = cv::imread(filename);

			std::cout << "Loaded: " << filename << " (" << temp.rows << "x" << temp.cols << ")" << std::endl;
			std::vector<cv::KeyPoint> kp;
			detector->detect(temp, kp);
			extractor->compute(temp, kp, desc);
			trainer->add(desc);
		}
		std::cout << "Clustering...";
		vocabulary = trainer->cluster();
		std::cout << "Done!" << std::endl;

		// Save the vocabulary
		if (saveTrainedVocab)
		{
			cv::FileStorage fs(vocabPath, cv::FileStorage::WRITE);
			fs << "vocabulary" << vocabulary;
			fs.release();
		}

		return vocabulary;
	}

	std::map<std::string, cv::SVM> trainSVMs(const std::string& trainingPath,
											 const cv::Ptr<cv::FeatureDetector> detector,
											 cv::Ptr<cv::BOWImgDescriptorExtractor> imgExtractor,
											 const bool lookForSavedSVMs = true,
											 const bool saveTrainedSVMs = true)
	{
		std::map<std::string, cv::SVM> svms;
		std::map<std::string, cv::Mat> posFeatures, negFeatures;
		std::vector<std::string> categories;

		boost::filesystem3::recursive_directory_iterator trainDir(trainingPath);

		cv::SVMParams svmParams;
		svmParams.kernel_type = cv::SVM::RBF;

		// Load categories and positive samples
		std::string currentCategory;
		for(boost::filesystem3::recursive_directory_iterator end_iter; trainDir != end_iter; ++trainDir)
		{
			if(trainDir.level() == 0)
			{
				// Get category name from name of the folder
				currentCategory = (trainDir->path()).filename().string();
				categories.push_back(currentCategory);

				svms.insert(std::pair<std::string, cv::SVM>(currentCategory, cv::SVM()));
				std::cerr << currentCategory << std::endl;

				if (lookForSavedSVMs && boost::filesystem3::exists(trainingPath + currentCategory + "/svm.xml"))
				{
					svms[currentCategory].load((trainingPath + currentCategory + "/svm.xml").c_str());
					std::cerr << svms[currentCategory].get_support_vector_count() << std::endl;
				}
				else
				{
					posFeatures.insert(std::pair<std::string, cv::Mat>(currentCategory, cv::Mat()));
					negFeatures.insert(std::pair<std::string, cv::Mat>(currentCategory, cv::Mat()));
				}
			}
			else
			{
				if ((trainDir->path()).filename().string() == "Thumbs.db") {continue;}
				if ((trainDir->path()).filename().string() == "svm.xml") {continue;}
				if (svms[currentCategory].get_support_vector_count() > 0) {continue;} // SVM already trained.

				// File name with path
				std::string filename = trainDir->path().parent_path().string() + std::string("/")
						+ (trainDir->path()).filename().string();

				cv::Mat tmp = cv::imread(filename);
				cv::Mat desc;

				// Compute the actual feature
				std::vector<cv::KeyPoint> kp;
				detector->detect(tmp, kp);
				imgExtractor->compute(tmp, kp, desc);

				std::cout << "Creating feature: " << filename << " (" << desc.rows << "x" << desc.cols << ")" << std::endl;

				posFeatures[currentCategory].push_back(desc);
			}
		}

		// Create negative samples from the positive samples of other classes
		for (std::string category : categories)
		{
			if (svms[category].get_support_vector_count() > 0) {continue;} // SVM already trained.

			for (std::map<std::string, cv::Mat>::iterator iter = posFeatures.begin();
				 iter != posFeatures.end(); ++iter)
			{
				if (iter->first != category)
				{
					std::cout << "Adding negative examples: " << iter->first << std::endl;
					if (negFeatures[category].rows == 0)
					{
						negFeatures[category] = iter->second;
					}
					else
					{
						cv::vconcat(negFeatures[category],
									iter->second,
									negFeatures[category]);
					}
				}
				else
				{
					std::cout << "Not Adding " << iter->first << std::endl;
				}
			}
		}

		// Train SVMs for each class
		for (std::string category : categories)
		{
			if (svms[category].get_support_vector_count() > 0) {continue;} // SVM already trained.

			cv::Mat trainData;
			cv::Mat trainLabels;

			std::cout << "Creating +/- training data for " << category << "...";
			cv::vconcat(posFeatures[category],
						negFeatures[category],
						trainData);
			cv::vconcat(cv::Mat::ones(posFeatures[category].rows, 1, CV_32S), // Positive training data has label 1
						cv::Mat::zeros(negFeatures[category].rows, 1, CV_32S), // Negative training data has label 0
						trainLabels);

			svms[category].train(trainData, trainLabels, cv::Mat(), cv::Mat(), svmParams);
			std::cout << "Done." << std::endl;
		}

		// Save svms
		if (saveTrainedSVMs)
		{
			for (std::map<std::string, cv::SVM>::iterator iter = svms.begin();
				 iter != svms.end(); ++iter)
			{
				std::string svmFilename = BASE_PATH + TRAINING_PATH + iter->first + "/svm.xml";
				iter->second.save(svmFilename.c_str());
			}
		}

		return svms;

	}

	void TestCluster()
	{
		cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector());
		cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SurfDescriptorExtractor());
		cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher());

		cv::Ptr<cv::BOWTrainer> trainer(new cv::BOWKMeansTrainer(BOW_FEATURE_SIZE));
		cv::Ptr<cv::BOWImgDescriptorExtractor> imgExtractor(new cv::BOWImgDescriptorExtractor(extractor, matcher));


		boost::filesystem3::directory_iterator templateDir(BASE_PATH + TEMPLATES_PATH);

		cv::Mat vocabulary = buildVocabulary(templateDir, detector, extractor, trainer, false);
		std::cout << "Vocab: " << " (" << vocabulary.rows << "x" << vocabulary.cols << ")" << std::endl;

		imgExtractor->setVocabulary(vocabulary);

		std::map<std::string, cv::SVM> svms = trainSVMs(BASE_PATH + TRAINING_PATH, detector, imgExtractor, false);

		// Do the testing set
		boost::filesystem3::directory_iterator testIter(BASE_PATH + TESTING_PATH);
		for (boost::filesystem3::directory_iterator end; testIter != end; ++testIter)
		{
			// Compute BoW feature for image
			std::string filename = testIter->path().parent_path().string() + std::string("/")
					+ (testIter->path()).filename().string();

			cv::Mat tmp = cv::imread(filename);
			cv::Mat desc;

			// Compute the actual feature
			std::vector<cv::KeyPoint> kp;
			detector->detect(tmp, kp);
			imgExtractor->compute(tmp, kp, desc);

			std::cout << (testIter->path()).filename().string() << ":" << std::endl;

			// Classify it
			for (std::map<std::string, cv::SVM>::iterator iter = svms.begin();
				 iter != svms.end(); ++iter)
			{
				std::cout << "\t" <<iter->first << ": " << iter->second.predict(desc, true) << std::endl;
			}


		}
	}
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBoW);
