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

import cc.mallet.pipe.Pipe;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.Instance;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.conversion.Cast;

import java.io.Serializable;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class VectorToTokensPipe extends Pipe implements Serializable {
   private static final long serialVersionUID = 1L;
   private final MalletVectorizer encoder;

   public VectorToTokensPipe(MalletVectorizer encoder) {
      this.encoder = encoder;
   }

   @Override
   public Instance pipe(Instance inst) {
      Example vector = Cast.as(inst.getData());
      List<Feature> features = vector.getFeatures();
      String[] names = new String[features.size()];
      double[] values = new double[features.size()];
      for (int i = 0; i < features.size(); i++) {
         Feature f = features.get(i);
         names[i] = f.getName();
         values[i] = f.getValue();
      }
      inst.setData(new FeatureVector(encoder.alphabet, names, values));
      return inst;
   }


}//END OF VectorToTokensPipe
