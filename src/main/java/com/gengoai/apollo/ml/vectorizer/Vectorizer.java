/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Pipeline;
import com.gengoai.apollo.ml.data.ExampleDataset;

import java.io.Serializable;

/**
 * <p>Vectorizers  are responsible for transforming instance {@link Example}s into {@link NDArray}s. Note: That
 * vectorizer implementations should only handle instance input. {@link Example} subclasses should provide the logic for
 * combining instances in their {@link Example#transform(Pipeline)} method.</p>
 *
 * @author David B. Bracewell
 */
public interface Vectorizer extends Serializable {

   /**
    * Fits the vectorizer to the given {@link ExampleDataset}.
    *
    * @param dataset the dataset to fit the vectorizer on
    */
   void fit(ExampleDataset dataset);

   /**
    * Transforms an {@link com.gengoai.apollo.ml.Instance} into an {@link NDArray}.
    *
    * @param example the instance to transform
    * @return the NDArray output of vectorizing the given instance
    */
   NDArray transform(Example example);

}// END OF Encoder
