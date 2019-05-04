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

import com.gengoai.apollo.linear.DenseMatrix;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.Params;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.vectorizer.CountFeatureVectorizer;
import com.gengoai.conversion.Cast;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

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
   protected void fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters p = notNull(Cast.as(fitParameters, Parameters.class));
      org.apache.spark.mllib.feature.Word2Vec w2v = new org.apache.spark.mllib.feature.Word2Vec();
      w2v.setMinCount(1);
      w2v.setVectorSize(p.dimension.value());
      w2v.setLearningRate(p.learningRate.value());
      w2v.setNumIterations(p.maxIterations.value());
      w2v.setWindowSize(p.windowSize.value());
      w2v.setSeed(new Date().getTime());
      w2v.setMinCount(1);

      List<String> suffix = new ArrayList<>();
      for (int i = 0; i < getPipeline().featureVectorizer.size(); i++) {
         String feature = getPipeline().featureVectorizer.getString(i);
         int index = feature.indexOf('=');
         if (index > 0) {
            suffix.add(feature.substring(index + 1));
         } else {
            suffix.add(feature);
         }
      }
      this.getPipeline().featureVectorizer = new CountFeatureVectorizer(suffix, getPipeline()
                                                                                   .featureVectorizer.unknown());

      Word2VecModel model = w2v.fit(preprocessed.stream()
                                                .toDistributedStream()
                                                .map(e -> exampleToList(e, false))
                                                .getRDD());
      NDArray[] vectors = new NDArray[model.getVectors().size()];
      mapAsJavaMap(model.getVectors()).forEach((k, v) -> {
         int index = getPipeline().featureVectorizer.indexOf(k);
         if (index >= 0) {
            vectors[index] = new DenseMatrix(MatrixFunctions.floatToDouble(new FloatMatrix(1, v.length, v)));
            vectors[index].setLabel(k);
         } else {
            System.out.println("-1: " + k);
         }
      });
      this.vectorIndex = new DefaultVectorIndex(vectors);
   }

   @Override
   public Word2Vec.Parameters getFitParameters() {
      return new Parameters();
   }

   /**
    * FitParameters for Word2Vec
    */
   public static class Parameters extends EmbeddingFitParameters<Parameters> {
      /**
       * The Learning rate.
       */
      public final Parameter<Double> learningRate = parameter(Params.Optimizable.learningRate, 0.025);
      /**
       * The Max iterations.
       */
      public final Parameter<Integer> maxIterations = parameter(Params.Optimizable.maxIterations, 1);
   }

}//END OF Word2Vec
