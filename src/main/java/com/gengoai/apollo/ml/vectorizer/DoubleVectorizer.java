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

import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.json.JsonEntry;

import java.lang.reflect.Type;

/**
 * @author David B. Bracewell
 */
public class DoubleVectorizer implements Vectorizer<Double> {
   private static final long serialVersionUID = 1L;

   public static DoubleVectorizer fromJson(JsonEntry entry, Type... parameters) {
      DoubleVectorizer vectorizer = new DoubleVectorizer();
      Validation.checkState(entry.getStringProperty("class").equalsIgnoreCase(DoubleVectorizer.class.getName()));
      return vectorizer;
   }

   @Override
   public Double decode(double value) {
      return value;
   }

   @Override
   public double encode(Double value) {
      return value;
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public NDArray transform(Example example) {
      NDArray ndArray = NDArrayFactory.DEFAULT().zeros(1, example.size());
      for (int i = 0; i < example.size(); i++) {
         Example c = example.getExample(i);
         if (c.getLabel() instanceof CharSequence) {
            ndArray.set(0, i, Double.parseDouble(c.getLabelAsString()));
         } else {
            ndArray.set(0, i, c.hasLabel() ? c.getLabelAsDouble() : Double.NaN);
         }
      }
      return ndArray;
   }

   @Override
   public int size() {
      return 1;
   }

   public JsonEntry toJson() {
      return JsonEntry.object()
                      .addProperty("class", DoubleVectorizer.class);
   }

   @Override
   public String toString() {
      return "DoubleVectorizer";
   }

}//END OF DoubleVectorizer
