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

package com.gengoai.apollo.ml.sequence;

import cc.mallet.pipe.Pipe;
import cc.mallet.types.Instance;
import cc.mallet.types.Token;
import cc.mallet.types.TokenSequence;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.Feature;
import com.gengoai.conversion.Cast;

import java.io.Serializable;

/**
 * @author David B. Bracewell
 */
public class SequenceToTokenSequence extends Pipe implements Serializable {
   @Override
   public Instance pipe(Instance inst) {
      Example vector = Cast.as(inst.getData());
      TokenSequence tokens = new TokenSequence(vector.size());
      for (int i = 0; i < vector.size(); i++) {
         Example e = vector.getExample(i);
         Token token = new Token("W-" + i);
         for (Feature feature : e.getFeatures()) {
            token.setFeatureValue(feature.getName(), feature.getValue());
         }
         tokens.add(token);
      }
      inst.setData(tokens);
      return inst;
   }
}//END OF SequenceToTokenSequence
