/**
 * \file KazeDescriptorExtractor.h
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

#ifndef KAZEDESCRIPTOREXTRACTOR_H
#define KAZEDESCRIPTOREXTRACTOR_H

#include <opencv2/features2d/features2d.hpp>
#include <boost/shared_ptr.hpp>

#include <KAZE.h>

namespace cv
{

class KazeDescriptorExtractor : public DescriptorExtractor
{
public:
	KazeDescriptorExtractor();

	//virtual void read( const FileNode& );
	//virtual void write( FileStorage& ) const;
	virtual int descriptorSize() const { return -1; } // TODO: Find out what this actually is
	virtual int descriptorType() const { return -1; }

	typedef boost::shared_ptr<KAZE> KazePtr;

	void preloadEnvironment(KazePtr& e);

protected:
	virtual void computeImpl( const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors ) const;

	bool usePreloadedEnvironment;
	KazePtr environment;
};

}
#endif // KAZEDESCRIPTOREXTRACTOR_H
