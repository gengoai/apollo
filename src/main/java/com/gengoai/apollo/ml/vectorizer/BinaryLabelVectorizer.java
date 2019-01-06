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

import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.collection.Sets;

import java.util.Set;

/**
 * <p>Specialized vectorizer to encode binary classes (0/1).</p>
 *
 * @author David B. Bracewell
 */
public class BinaryLabelVectorizer extends StringVectorizer {
   private static final long serialVersionUID = 1L;
   private final String trueLabel;
   private final String falseLabel;

   /**
    * Instantiates a new Binary vectorizer using "true" and "false" as the labels.
    */
   public BinaryLabelVectorizer() {
      this("true", "false");
   }

   /**
    * Instantiates a new Binary vectorizer.
    *
    * @param trueLabel  the true label
    * @param falseLabel the false label
    */
   public BinaryLabelVectorizer(String trueLabel, String falseLabel) {
      super(true);
      this.trueLabel = trueLabel;
      this.falseLabel = falseLabel;
   }

   @Override
   public Set<String> alphabet() {
      return Sets.hashSetOf(falseLabel, trueLabel);
   }


   @Override
   public String decode(double value) {
      return value == 1.0
             ? trueLabel
             : falseLabel;
   }

   @Override
   public double encode(String value) {
      return value.equals(trueLabel)
             ? 1.0
             : 0.0;
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public int size() {
      return 2;
   }

}//END OF BinaryLabelVectorizer
