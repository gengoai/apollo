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

package com.gengoai.apollo.ml.embedding;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.params.ParamMap;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import org.apache.spark.mllib.feature.Word2VecModel;

import static scala.collection.JavaConversions.mapAsJavaMap;

/**
 * <p>Wrapper around Spark's Word2Vec embedding model.</p>
 *
 * @author David B. Bracewell
 */
public class Word2Vec extends Embedding {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Word2Vec.
    *
    * @param preprocessors the preprocessors
    */
   public Word2Vec(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   @Override
   protected void fitPreprocessed(Dataset preprocessed, ParamMap p) {
      org.apache.spark.mllib.feature.Word2Vec w2v = new org.apache.spark.mllib.feature.Word2Vec();
      w2v.setMinCount(1);
      w2v.setVectorSize(p.get(dimension));
      w2v.setLearningRate(p.get(learningRate));
      w2v.setNumIterations(p.get(maxIterations));
      w2v.setWindowSize(p.get(windowSize));
      w2v.setMinCount(1);
      Word2VecModel model = w2v.fit(preprocessed.stream()
                                                .toDistributedStream()
                                                .map(e -> exampleToList(e, false))
                                                .getRDD());
      NDArray[] vectors = new NDArray[model.getVectors().size()];
      mapAsJavaMap(model.getVectors()).forEach((k, v) -> {
         int index = getPipeline().featureVectorizer.indexOf(k);
         vectors[index] = NDArrayFactory.rowVector(v);
         vectors[index].setLabel(k);
      });
      this.vectorIndex = new DefaultVectorIndex(vectors);
   }

   @Override
   public ParamMap getFitParameters() {
      return new ParamMap(
         dimension.set(100),
         learningRate.set(0.025),
         maxIterations.set(1),
         windowSize.set(5),
         verbose.set(false)
      );
   }


}//END OF Word2Vec
