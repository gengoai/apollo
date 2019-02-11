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

import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.store.VSBuilder;
import com.gengoai.apollo.linear.store.VectorStore;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.conversion.Cast;
import org.apache.spark.mllib.feature.Word2VecModel;

import java.util.Date;

import static com.gengoai.Validation.notNull;
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
   protected VectorStore fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters p = notNull(Cast.as(fitParameters, Parameters.class));
      org.apache.spark.mllib.feature.Word2Vec w2v = new org.apache.spark.mllib.feature.Word2Vec();
      w2v.setMinCount(1);
      w2v.setVectorSize(p.dimension);
      w2v.setLearningRate(p.learningRate);
      w2v.setNumIterations(p.numIterations);
      w2v.setWindowSize(p.windowSize);
      w2v.setSeed(p.randomSeed);
      w2v.setMinCount(1);
      Word2VecModel model = w2v.fit(preprocessed.stream()
                                                .toDistributedStream()
                                                .map(e -> exampleToList(e, false))
                                                .getRDD());
      VSBuilder builder = p.vectorStoreBuilder.get();
      mapAsJavaMap(model.getVectors()).forEach((k, v) -> builder.add(k, NDArrayFactory.rowVector(v)));
      return builder.build();
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return new Parameters();
   }

   /**
    * FitParameters for Word2Vec
    */
   public static class Parameters extends EmbeddingFitParameters {
      /**
       * The number of iterations to train the embeddings (default = 1)
       */
      public int numIterations = 1;
      /**
       * The learning rate (default = 0.025)
       */
      public double learningRate = 0.025;
      /**
       * The random seed (default = current time)
       */
      public long randomSeed = new Date().getTime();
      /**
       * The window size (default = 5)
       */
      public int windowSize = 5;
      /**
       * The dimension of the embeddings (default = 100)
       */
      public int dimension = 100;
   }

}//END OF Word2Vec
