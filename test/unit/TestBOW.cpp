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

#include <KazeFeatureDetector.h>
#include <KazeDescriptorExtractor.h>

#include "voc_utils.h"
#include "caltech101_utils.h"

const int BOW_FEATURE_SIZE = 32;
const std::string BASE_PATH = "/home/arprice/Desktop/bow/";
//const std::string TEMPLATES_PATH = "templates/";
//const std::string TRAINING_PATH = "training/";
const std::string TESTING_PATH = "testing/";

typedef std::map<std::string, std::vector<std::string> > TrainingSet;

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

	// Creates a 1/n subset of the data with offset o
	TrainingSet fold(TrainingSet& ts, int n, int o = 0, bool removeFromOriginal = false)
	{
		TrainingSet newSet;
		for (TrainingSet::iterator catIter = ts.begin();
			 catIter != ts.end(); ++catIter)
		{
			int count = 0;
			std::set<std::vector<std::string>::iterator> rmIters;
			newSet.insert(std::pair<std::string, std::vector<std::string> >(catIter->first, std::vector<std::string>()));
			//for (std::string name : catIter->second)
			for (std::vector<std::string>::iterator fileIter = catIter->second.begin();
				 fileIter != catIter->second.end(); ++fileIter)
			{
				if (count % n == o)
				{
					std::string name = *fileIter;
					newSet[catIter->first].push_back(name);
					if (removeFromOriginal)
					{
						rmIters.insert(fileIter);
					}
				}
				++count;
			}

			if (removeFromOriginal)
			{
				for (std::set<std::vector<std::string>::iterator>::iterator fileIter = rmIters.begin();
					 fileIter != rmIters.end(); ++fileIter)
				{
					catIter->second.erase(*fileIter);
				}
			}
		}
		return newSet;
	}

	void setSVMTrainAutoParams( cv::ParamGrid& c_grid, cv::ParamGrid& gamma_grid,
								cv::ParamGrid& p_grid, cv::ParamGrid& nu_grid,
								cv::ParamGrid& coef_grid, cv::ParamGrid& degree_grid )
	{
		c_grid = cv::SVM::get_default_grid(cv::SVM::C);

		gamma_grid = cv::SVM::get_default_grid(cv::SVM::GAMMA);

		p_grid = cv::SVM::get_default_grid(cv::SVM::P);
		p_grid.step = 0;

		nu_grid = cv::SVM::get_default_grid(cv::SVM::NU);
		nu_grid.step = 0;

		coef_grid = cv::SVM::get_default_grid(cv::SVM::COEF);
		coef_grid.step = 0;

		degree_grid = cv::SVM::get_default_grid(cv::SVM::DEGREE);
		degree_grid.step = 0;
	}

	cv::Mat buildVocabulary(TrainingSet& t,
							const cv::Ptr<cv::FeatureDetector> detector,
							const cv::Ptr<cv::DescriptorExtractor> extractor,
							cv::Ptr<cv::BOWTrainer> trainer,
							const bool lookForSavedVocab = true,
							const bool saveTrainedVocab = true,
							const bool validateKPs = true)
	{
		cv::Mat vocabulary;
		std::string vocabPath = BASE_PATH + "/vocab.xml";

		// Look for pretrained vocabulary
		if (lookForSavedVocab && boost::filesystem3::exists(vocabPath))
		{
			std::cout << "Loading filesystem vocabulary: " << vocabPath << std::endl;
			cv::FileStorage fs(vocabPath, cv::FileStorage::READ);
			fs["vocabulary"] >> vocabulary;
			fs.release();

			return vocabulary;
		}

		// Use templates to train a vocabulary
		cv::Mat desc;
		//for (boost::filesystem3::directory_iterator end; iter != end; ++iter)
		for (TrainingSet::iterator catIter = t.begin();
			 catIter != t.end(); ++catIter)
		{
			for (std::string name : catIter->second)
			{
				//std::string filename = VOC_FILE_STRUCTURE.imgDir() + "/" + name + ".jpg";
				std::string filename = name;
				cv::Mat temp = cv::imread(filename);

				assert(temp.rows > 0 && temp.cols > 0);

				std::cout << "Loaded: " << filename << " (" << temp.rows << "x" << temp.cols << ")" << std::endl;
				std::vector<cv::KeyPoint> kp;
				detector->detect(temp, kp);
				if (validateKPs)
				{
					validateKeypoints(kp, getMask(name));
				}
				if (kp.size() == 0) {continue;}
				extractor->compute(temp, kp, desc);

				assert(kp.size() > 0);

//				draw_keypoints(temp, kp);
//				cv::imshow("test", temp);
//				cv::imwrite("/home/arprice/Desktop/kaze" + name + ".jpg", temp);
//				cv::waitKey();
				trainer->add(desc);
			}
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

	std::map<std::string, cv::SVM> trainSVMs(TrainingSet& t,
											 const cv::Ptr<cv::FeatureDetector> detector,
											 cv::Ptr<cv::BOWImgDescriptorExtractor> imgExtractor,
											 const bool lookForSavedSVMs = true,
											 const bool saveTrainedSVMs = true,
											 const bool validateKPs = true)
	{
		std::map<std::string, cv::SVM> svms;
		std::map<std::string, cv::Mat> posFeatures, negFeatures;
		std::vector<std::string> categories;

		cv::SVMParams svmParams;
		svmParams.svm_type = CvSVM::C_SVC;
		svmParams.kernel_type = cv::SVM::RBF;
		//svmParams.kernel_type = cv::SVM::LINEAR;

		// Load categories and positive samples
		for (TrainingSet::iterator catIter = t.begin();
			 catIter != t.end(); ++catIter)
		{
			std::string currentCategory = catIter->first;
			categories.push_back(currentCategory);

			svms.insert(std::pair<std::string, cv::SVM>(currentCategory, cv::SVM()));
			std::cerr << currentCategory << std::endl;

			if (lookForSavedSVMs && boost::filesystem3::exists(BASE_PATH + "/" + currentCategory + "_svm.xml"))
			{
				svms[currentCategory].load((BASE_PATH + "/" + currentCategory + "_svm.xml").c_str());
				std::cerr << svms[currentCategory].get_support_vector_count() << std::endl;
			}
			else
			{
				posFeatures.insert(std::pair<std::string, cv::Mat>(currentCategory, cv::Mat()));
				negFeatures.insert(std::pair<std::string, cv::Mat>(currentCategory, cv::Mat()));
			}

			for (std::string name : catIter->second)
			{
				if (svms[currentCategory].get_support_vector_count() > 0) {continue;} // SVM already trained.

				// File name with path
				//std::string filename = VOC_FILE_STRUCTURE.imgDir() + "/" + name + ".jpg";
				std::string filename = name;

				cv::Mat tmp = cv::imread(filename);
				cv::Mat desc;

				// Compute the actual feature
				std::vector<cv::KeyPoint> kp;
				detector->detect(tmp, kp);
				if (validateKPs)
				{
					validateKeypoints(kp, getMask(name));
				}
				if (kp.size() == 0) {continue;}
				imgExtractor->compute(tmp, kp, desc);

				std::cout << "Creating feature: " << filename << " (" << desc.rows << "x" << desc.cols << ")" << std::endl;

				posFeatures[currentCategory].push_back(desc);
			}
		}

		// Create negative samples from the positive samples of other classes
		for (std::string category : categories)
		{
			if (svms[category].get_support_vector_count() > 0) {continue;} // SVM already trained.

			std::cerr << "Creating - examples for " << category << std::endl;
//			if (category != "BACKGROUND_Google")
//			{
//				negFeatures[category] = posFeatures["BACKGROUND_Google"];
//			}

			{
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
							if (iter->first != "BACKGROUND_Google")
							{
								cv::vconcat(negFeatures[category],
											iter->second.rowRange(1,3),
											negFeatures[category]);
							}
							else
							{
								cv::vconcat(negFeatures[category],
											iter->second,
											negFeatures[category]);
							}
						}
					}
					else
					{
						std::cout << "Not Adding " << iter->first << std::endl;
					}
				}
			}
		}

		bool fireOnce = true;
		// Train SVMs for each class
		for (std::string category : categories)
		{
			if (svms[category].get_support_vector_count() > 0) {continue;} // SVM already trained.

			cv::Mat trainData;
			cv::Mat trainLabels;

			assert(posFeatures[category].data != NULL && negFeatures[category].data != NULL);
			assert(posFeatures[category].cols == negFeatures[category].cols);

			std::cout << "Creating +/- training data for " << category << "...";
			cv::vconcat(posFeatures[category],
						negFeatures[category],
						trainData);
			cv::vconcat(cv::Mat::ones(posFeatures[category].rows, 1, CV_32S), // Positive training data has label 1
						-cv::Mat::ones(negFeatures[category].rows, 1, CV_32S), // Negative training data has label -1
						trainLabels);

			cv::ParamGrid c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid;
			setSVMTrainAutoParams( c_grid, gamma_grid,  p_grid, nu_grid, coef_grid, degree_grid );
			svms[category].train_auto( trainData, trainLabels, cv::Mat(), cv::Mat(), svmParams, 10, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid );

			//svms[category].train(trainData, trainLabels, cv::Mat(), cv::Mat(), svmParams);
			std::cout << "Done." << std::endl;

			if (fireOnce)
			{
				fireOnce = false;
				std::cerr << trainData << std::endl;
				std::cerr << trainLabels << std::endl;
			}
		}

		// Save svms
		if (saveTrainedSVMs)
		{
			for (std::map<std::string, cv::SVM>::iterator iter = svms.begin();
				 iter != svms.end(); ++iter)
			{
				std::string svmFilename = BASE_PATH + "/" + iter->first + "_svm.xml";
				iter->second.save(svmFilename.c_str());
			}
		}

		return svms;

	}

	cv::Mat testSVMs(TrainingSet& t,
					 const cv::Ptr<cv::FeatureDetector> detector,
					 cv::Ptr<cv::BOWImgDescriptorExtractor> imgExtractor,
					 const std::map<std::string, cv::SVM>& svms)
	{
		cv::Mat confusion = cv::Mat::zeros(t.size()-1, t.size()-1, CV_32FC1);

		std::map<std::string, int> catMap;
		int catCount = 0;

		for (TrainingSet::iterator catIter = t.begin();
			 catIter != t.end(); ++catIter)
		{
			catMap.insert(std::pair<std::string, int>(catIter->first, catCount++));
			std::cout << catIter->first << "','";
		}
		std::cout << std::endl;

		for (TrainingSet::iterator catIter = t.begin();
			 catIter != t.end(); ++catIter)
		{
			if (catIter->first == "BACKGROUND_Google") { continue; }

			for (std::string name : catIter->second)
			{
				std::string filename = name;
				cv::Mat tmp = cv::imread(filename);

				cv::Mat desc;

				// Compute the actual feature
				std::vector<cv::KeyPoint> kp;
				detector->detect(tmp, kp);
				imgExtractor->compute(tmp, kp, desc);

				//float maxPrediction = -100;
				float minPrediction =  100;

				//std::string maxClass;
				std::string minClass;

				// Classify it
				for (std::map<std::string, cv::SVM>::const_iterator iter = svms.begin();
					 iter != svms.end(); ++iter)
				{
					if (iter->first == "BACKGROUND_Google") {continue;}
					float prediction = iter->second.predict(desc, true);
					//if (prediction > maxPrediction) { maxPrediction = prediction; maxClass = iter->first; }
					if (prediction < minPrediction) { minPrediction = prediction; minClass = iter->first; }
					//std::cout << "\t" <<iter->first << ": " << prediction << std::endl;
				}
				bool correct = catIter->first == minClass;
				if (correct)
				{
					std::cout << "^^^^^^^^Correct: " << minClass << std::endl;
				}
				else
				{
					std::cout << "~~~~~~~~Incorrect: " << minClass << std::endl;
				}

				// Increment this element in the confusion matrix
				confusion.at<float>(catMap[catIter->first] * confusion.cols + catMap[minClass]) += 1;
				std::cout << std::endl;
				//cv::imwrite(BASE_PATH + TESTING_PATH + minClass + std::to_string(rand()%1000) + ".jpg", tmp);
			}
		}

		return confusion;
	}

	//std::string getConfusionLaTeX(cv::Mat confusion, )

	void TestCluster()
	{
		cv::Ptr<cv::FeatureDetector> detector(new cv::SiftFeatureDetector());
		cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SiftDescriptorExtractor());
		cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher());

		cv::Ptr<cv::BOWTrainer> trainer(new cv::BOWKMeansTrainer(BOW_FEATURE_SIZE));
		cv::Ptr<cv::BOWImgDescriptorExtractor> imgExtractor(new cv::BOWImgDescriptorExtractor(extractor, matcher));

		TrainingSet collections, templates, tests;
		//readVOCLists(VOC_FILE_STRUCTURE.setDir(), collections, 5);
		loadCT101("/media/Data/101_ObjectCategories/", collections, 20, 100);

		templates = fold(collections, 5, 0, true);

		tests = fold(collections, 10, 0, true);

		cv::Mat vocabulary = buildVocabulary(templates, detector, extractor, trainer, true, true, false);
		std::cout << "Vocab: " << " (" << vocabulary.rows << "x" << vocabulary.cols << ")" << std::endl;

		imgExtractor->setVocabulary(vocabulary);

		std::map<std::string, cv::SVM> svms = trainSVMs(collections, detector, imgExtractor, true, true, false);

		cv::Mat confusion = testSVMs(tests, detector, imgExtractor, svms);

		//cv::Mat confusion(counts.size(), counts.type());
		for (int i = confusion.rows - 1; i >= 0; i--)
		{
			//cv::normalize(counts.row(i), cv::_OutputArray(confusion.ptr(i), confusion.cols));
			cv::normalize(confusion.row(i), confusion.row(i));
		}

		std::cout << confusion << std::endl;

		confusion *= 255.0;
		cv::cvtColor(confusion, confusion, CV_GRAY2RGB);
		confusion.convertTo(confusion, CV_8UC3);

		cv::namedWindow("confusion", CV_WINDOW_NORMAL);
		cv::imwrite(BASE_PATH + TESTING_PATH + "confusion.jpg", confusion);

		cv::imshow("confusion", confusion);
		cv::waitKey();

		// Do the testing set
//		boost::filesystem3::directory_iterator testIter(BASE_PATH + TESTING_PATH);
//		for (boost::filesystem3::directory_iterator end; testIter != end; ++testIter)
//		{
//			// Compute BoW feature for image
//			std::string filename = testIter->path().parent_path().string() + std::string("/")
//					+ (testIter->path()).filename().string();

//			cv::Mat tmp = cv::imread(filename);
//			cv::Mat desc;

//			// Compute the actual feature
//			std::vector<cv::KeyPoint> kp;
//			detector->detect(tmp, kp);
//			imgExtractor->compute(tmp, kp, desc);

//			std::cout << (testIter->path()).filename().string() << ":" << std::endl;

//			// Classify it
//			for (std::map<std::string, cv::SVM>::iterator iter = svms.begin();
//				 iter != svms.end(); ++iter)
//			{
//				std::cout << "\t" <<iter->first << ": " << iter->second.predict(desc, true) << std::endl;
//			}
//		}
	}
};

CPPUNIT_TEST_SUITE_REGISTRATION(TestBoW);
