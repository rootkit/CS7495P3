/**
 * \file caltech101_utils.cpp
 * \brief
 *
 * \author Andrew Price
 * \date 12 3, 2013
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

#include "caltech101_utils.h"

#include <boost/filesystem.hpp>

void loadCT101(const std::string directory, TrainingSet& collections, int numCategories, int numExamples)
{
	boost::filesystem3::recursive_directory_iterator trainDir(directory);
	int numCat = 0;
	int numEx = 0;

	// Load categories and positive samples
	std::string currentCategory;
	for(boost::filesystem3::recursive_directory_iterator end_iter; trainDir != end_iter; ++trainDir)
	{
		if(trainDir.level() == 0)
		{
			// Get category name from name of the folder
			currentCategory = (trainDir->path()).filename().string();

			if (numCategories > 0 && numCat >= numCategories) {break;}
			++numCat;

			collections.insert(std::pair<std::string, std::vector<std::string> >(currentCategory, std::vector<std::string>()));

			numEx = 0;
		}
		else
		{
			if (numExamples < 0 || numEx < numExamples)
			{
				// File name with path
				std::string filename = trainDir->path().parent_path().string() + "/" + (trainDir->path()).filename().string();

				collections[currentCategory].push_back(filename);
				++numEx;
			}
		}
	}
}
