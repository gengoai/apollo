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
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.conversion.Cast;
import com.gengoai.stream.SparkStream;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import static com.gengoai.Validation.notNull;
import static com.gengoai.apollo.linear.SparkLinearAlgebra.sparkSVD;
import static com.gengoai.apollo.linear.SparkLinearAlgebra.toMatrix;

/**
 * @author David B. Bracewell
 */
public class LSA extends TopicModel {
   private static final long serialVersionUID = 1L;
   private List<LSATopic> topics = new ArrayList<>();

   @Override
   public double[] estimate(Example example) {
      double[] scores = new double[topics.size()];
      NDArray vector = encodeAndPreprocess(example);
      for (int i = 0; i < topics.size(); i++) {
         double score = vector.scalarDot(getTopic(i).vector);
         scores[i] = score;
      }
      return scores;
   }

   @Override
   protected TopicModel fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters parameters = notNull(Cast.as(fitParameters, Parameters.class));
      //Create document x word matrix
      SparkStream<Vector> stream = new SparkStream<Vector>(preprocessed.stream()
                                                                       .map(this::encode)
                                                                       .map(n -> new DenseVector(n.toDoubleArray())))
                                      .cache();
      RowMatrix mat = new RowMatrix(stream.getRDD().rdd());
      //since we have document x word, V is the word x component matrix
      // U = document x component, E = singular components, V = word x component
      // Transpose V to get component (topics) x words
      NDArray topicMatrix = toMatrix(sparkSVD(mat, parameters.K).V().transpose());
      for (int i = 0; i < parameters.K; i++) {
         topics.add(new LSATopic(NDArrayFactory.columnVector(topicMatrix.getVector(i, Axis.ROW).toDoubleArray())));
      }
      return this;
   }

   @Override
   public FitParameters getDefaultFitParameters() {
      return new Parameters();
   }

   public static class Parameters extends FitParameters {
      public int K = 100;
   }

   @Override
   public NDArray getTopicDistribution(String feature) {
      int i = (int) getFeatureVectorizer().encode(feature);
      if (i == -1) {
         return NDArrayFactory.rowVector(new double[topics.size()]);
      }
      double[] dist = new double[topics.size()];
      for (int i1 = 0; i1 < topics.size(); i1++) {
         dist[i1] = getTopic(i1).vector.get(i);
      }
      return NDArrayFactory.rowVector(dist);
   }

   @Override
   public LSATopic getTopic(int topic) {
      return topics.get(topic);
   }

   @Override
   public int getNumberOfTopics() {
      return topics.size();
   }

   public class LSATopic implements Topic, Serializable {
      private static final long serialVersionUID = 1L;
      private final NDArray vector;

      public LSATopic(NDArray vector) {
         this.vector = vector;
      }


      @Override
      public Counter<String> featureDistribution() {
         Counter<String> c = Counters.newCounter();
         vector.forEachSparse(e -> c.set(getFeatureVectorizer().decode(e.getIndex()), e.getValue()));
         return c;
      }
   }


}//END OF LSA
