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

package com.gengoai.apollo.ml.topic;

import com.gengoai.apollo.linear.Axis;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.conversion.Cast;
import com.gengoai.stream.SparkStream;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

import java.util.ArrayList;
import java.util.List;

import static com.gengoai.apollo.linear.SparkLinearAlgebra.sparkSVD;
import static com.gengoai.apollo.linear.SparkLinearAlgebra.toMatrix;

/**
 * <p>Distributed version of <a href="https://en.wikipedia.org/wiki/Latent_semantic_analysis">Latent Semantic
 * Analysis</a> using Apache Spark. Documents are represented by examples and words are by features in the Example.</p>
 *
 * @author David B. Bracewell
 */
public class LSA extends TopicModel {
   private static final long serialVersionUID = 1L;
   private List<NDArray> topicVectors = new ArrayList<>();

   /**
    * Instantiates a new Lsa.
    *
    * @param preprocessors the preprocessors
    */
   public LSA(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Lsa.
    *
    * @param modelParameters the model parameters
    */
   public LSA(DiscretePipeline modelParameters) {
      super(modelParameters);
   }

   @Override
   public double[] estimate(Example example) {
      double[] scores = new double[topics.size()];
      NDArray vector = example.preprocessAndTransform(getPipeline());
      for (int i = 0; i < topics.size(); i++) {
         double score = vector.scalarDot(topicVectors.get(i));
         scores[i] = score;
      }
      return scores;
   }

   @Override
   protected void fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters parameters = Cast.as(fitParameters);
      //Create document x word matrix
      SparkStream<Vector> stream = new SparkStream<Vector>(preprocessed.asVectorStream(getPipeline())
                                                                       .map(n -> new DenseVector(n.toDoubleArray())))
                                      .cache();
      RowMatrix mat = new RowMatrix(stream.getRDD().rdd());
      //since we have document x word, V is the word x component matrix
      // U = document x component, E = singular components, V = word x component
      // Transpose V to get component (topics) x words
      NDArray topicMatrix = toMatrix(sparkSVD(mat, parameters.K).V().transpose());
      for (int i = 0; i < parameters.K; i++) {
         Counter<String> featureDist = Counters.newCounter();
         NDArray dist = NDArrayFactory.columnVector(topicMatrix.getVector(i, Axis.ROW).toDoubleArray());
         dist.forEachSparse(
            e -> featureDist.set(getPipeline().featureVectorizer.getString(e.getIndex()), e.getValue()));
         topics.add(new Topic(i, featureDist));
         topicVectors.add(dist);
      }
   }

   @Override
   public LSA.Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   /**
    * The type Parameters.
    */
   public static class Parameters extends FitParameters {
      /**
       * The K.
       */
      public int K = 100;
   }

   @Override
   public NDArray getTopicDistribution(String feature) {
      int i = getPipeline().featureVectorizer.indexOf(feature);
      if (i == -1) {
         return NDArrayFactory.rowVector(new double[topics.size()]);
      }
      double[] dist = new double[topics.size()];
      for (int i1 = 0; i1 < topics.size(); i1++) {
         dist[i1] = topicVectors.get(i1).get(i);
      }
      return NDArrayFactory.rowVector(dist);
   }


}//END OF LSA
