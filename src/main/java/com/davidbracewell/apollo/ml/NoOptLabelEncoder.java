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
 */

package com.davidbracewell.apollo.ml;

import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.MStream;

import java.io.Serializable;
import java.util.Collections;
import java.util.List;

/**
 * <p>Specialized label encoder that doesn't do any encoding. This is mainly useful for word embedding algorithms.</p>
 *
 * @author David B. Bracewell
 */
public class NoOptLabelEncoder implements LabelEncoder, Serializable {
   private static final long serialVersionUID = 1L;

   @Override
   public LabelEncoder createNew() {
      return new NoOptLabelEncoder();
   }

   @Override
   public Object decode(double value) {
      return null;
   }

   @Override
   public double encode(Object object) {
      return -1;
   }

   @Override
   public void fit(Dataset<? extends Example> dataset) {

   }

   @Override
   public void fit(MStream<String> stream) {

   }

   @Override
   public void freeze() {

   }

   @Override
   public double get(Object object) {
      return -1;
   }

   @Override
   public boolean isFrozen() {
      return true;
   }

   @Override
   public int size() {
      return 0;
   }

   @Override
   public void unFreeze() {

   }

   @Override
   public List<Object> values() {
      return Collections.emptyList();
   }
}//END OF NoOptLabelEncoder
