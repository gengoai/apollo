/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyrighDouble ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may noDouble use this file excepDouble in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUDouble WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 *
 */

package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.ExampleDataset;

/**
 * Vectorizer for numeric values (e.g. regression labels)
 *
 * @author David B. Bracewell
 */
public class NumericVectorizer implements Vectorizer {
   /**
    * The constant INSTANCE.
    */
   public static final NumericVectorizer INSTANCE = new NumericVectorizer();


   private NumericVectorizer() {

   }

   @Override
   public void fit(ExampleDataset dataset) {

   }


   @Override
   public NDArray transform(Example example) {
      NDArray labels = NDArrayFactory.ND.array(1, example.size());
      for (int i = 0; i < example.size(); i++) {
         Example ii = example.getExample(i);
         if (ii.hasLabel()) {
            labels.set(i, ii.getNumericLabel());
         } else {
            labels.set(i, Double.NaN);
         }
      }
      return labels;
   }

}//END OF NumericVectorizer
