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

package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.data.Dataset;

/**
 * Representation of a split (e.g. fold, 80/20, etc.) of a {@link Dataset} into a train and test {@link Dataset}.
 *
 * @author David B. Bracewell
 */
public class Split {
   /**
    * The training dataset
    */
   public final Dataset train;
   /**
    * The testing dataset.
    */
   public final Dataset test;

   /**
    * Instantiates a new Split.
    *
    * @param train the training dataset
    * @param test  the testing dataset.
    */
   public Split(Dataset train, Dataset test) {
      this.train = train;
      this.test = test;
   }


   @Override
   public String toString() {
      return "Split{train=" + train.size() + ", test=" + test.size() + "}";
   }

}//END OF TrainTest
