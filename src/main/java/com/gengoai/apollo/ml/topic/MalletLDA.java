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

import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TargetStringToFeatures;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.topics.TopicInferencer;
import cc.mallet.types.Alphabet;
import cc.mallet.types.IDSorter;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import com.gengoai.SystemInfo;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.vectorizer.InstanceToTokenSequence;
import com.gengoai.apollo.ml.vectorizer.MalletVectorizer;
import com.gengoai.collection.counter.Counter;
import com.gengoai.collection.counter.Counters;
import com.gengoai.conversion.Cast;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.TreeSet;
import java.util.logging.Level;

/**
 * @author David B. Bracewell
 */
public class MalletLDA extends TopicModel {
   private static final long serialVersionUID = 1L;
   private volatile transient TopicInferencer inferencer;
   private SerialPipes pipes;
   private ParallelTopicModel topicModel;

   public MalletLDA(Preprocessor... preprocessors) {
      super(DiscretePipeline.unsupervised(new MalletVectorizer(new Alphabet()), preprocessors));
   }

   @Override
   public double[] estimate(Example example) {
      InstanceList instances = new InstanceList(pipes);
      instances.addThruPipe(
         new cc.mallet.types.Instance(getPipeline().preprocessorList.apply(example), "", null, null));
      return getInferencer().getSampledDistribution(instances.get(0), 800, 5, 100);
   }

   @Override
   protected void fitPreprocessed(Dataset preprocessed, FitParameters fitParameters) {
      Parameters p = Cast.as(fitParameters);
      if (p.verbose) {
         ParallelTopicModel.logger.setLevel(Level.INFO);
      } else {
         ParallelTopicModel.logger.setLevel(Level.OFF);
      }
      pipes = new SerialPipes(Arrays.asList(new TargetStringToFeatures(),
                                            new InstanceToTokenSequence(),
                                            new TokenSequence2FeatureSequence()));
      InstanceList trainingData = new InstanceList(pipes);
      preprocessed.forEach(i -> trainingData.addThruPipe(
         new Instance(i, i.getLabel() == null ? "" : i.getLabel().toString(), null, null)));
      topicModel = new ParallelTopicModel(p.K);
      topicModel.addInstances(trainingData);
      topicModel.setNumIterations(p.maxIterations);
      topicModel.setNumThreads(SystemInfo.NUMBER_OF_PROCESSORS - 1);
      topicModel.setBurninPeriod(p.burnIn);
      topicModel.setOptimizeInterval(p.optimizationInterval);
      topicModel.setSymmetricAlpha(p.symmetricAlpha);
      try {
         topicModel.estimate();
      } catch (IOException e) {
         throw new RuntimeException(e);
      }
   }

   @Override
   public MalletLDA.Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   private TopicInferencer getInferencer() {
      if (inferencer == null) {
         synchronized (this) {
            if (inferencer == null) {
               inferencer = topicModel.getInferencer();
               inferencer.setRandomSeed(1234);
            }
         }
      }
      return inferencer;
   }

   public int getNumberOfTopics() {
      return topicModel.numTopics;
   }

   @Override
   public Topic getTopic(int topic) {
      final Alphabet alphabet = pipes.getDataAlphabet();
      final ArrayList<TreeSet<IDSorter>> topicWords = topicModel.getSortedWords();
      double[][] termScores = topicModel.getTopicWords(true, true);
      Iterator iterator = topicWords.get(topic).iterator();
      IDSorter info;
      Counter<String> topicWordScores = Counters.newCounter();
      while (iterator.hasNext()) {
         info = (IDSorter) iterator.next();
         topicWordScores.set(alphabet.lookupObject(info.getID()).toString(), termScores[topic][info.getID()]);
      }
      return new MalletTopic(topicWordScores);
   }

   @Override
   public NDArray getTopicDistribution(String feature) {
      final Alphabet alphabet = pipes.getDataAlphabet();
      int index = alphabet.lookupIndex(feature, false);
      if (index == -1) {
         return NDArrayFactory.DEFAULT().zeros(topicModel.numTopics);
      }
      double[] dist = new double[topicModel.numTopics];
      double[][] termScores = topicModel.getTopicWords(true, true);
      for (int i = 0; i < topicModel.numTopics; i++) {
         dist[i] = termScores[i][index];
      }
      return NDArrayFactory.rowVector(dist);
   }

   public static class Parameters extends FitParameters {
      public int K = 100;
      public int burnIn = 500;
      public int maxIterations = 2000;
      public int optimizationInterval = 100;
      public boolean symmetricAlpha = false;
   }

   private class MalletTopic implements Topic, Serializable {
      private final Counter<String> dist;

      private MalletTopic(Counter<String> dist) {
         this.dist = dist;
      }

      @Override
      public Counter<String> featureDistribution() {
         return dist;
      }
   }

}//END OF MalletLDA
