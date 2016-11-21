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

package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.TrainTestSet;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.source.DenseCSVDataSource;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.apollo.ml.preprocess.transform.ZScoreTransform;
import com.davidbracewell.io.Resources;
import lombok.SneakyThrows;
import org.junit.Test;

import java.util.Random;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class ClassificationTest {

   private final ClassifierLearner learner;
   private final double targetAcc;
   private final double tolerance;
   private final boolean normalize;

   protected ClassificationTest(ClassifierLearner learner, double targetAcc, double tolerance, boolean normalize) {
      this.learner = learner;
      this.targetAcc = targetAcc;
      this.tolerance = tolerance;
      this.normalize = normalize;
   }

   @SneakyThrows
   public Dataset<Instance> getData() {
      DenseCSVDataSource irisData = new DenseCSVDataSource(Resources.fromClasspath(
         "com/davidbracewell/apollo/ml/iris.csv"), true);
      irisData.setClassIndex(4);
      return Dataset.classification()
                    .source(irisData)
                    .shuffle(new Random(1234));
   }

   @Test
   public void testClassification() throws Exception {
      TrainTestSet<Instance> trainTestSplits = getData().split(0.8);
      if (normalize) {
         trainTestSplits.preprocess(() -> PreprocessorList.create(
            new ZScoreTransform("sepal_length"),
            new ZScoreTransform("petal_length")));
      }

      trainTestSplits.forEach((train, test) -> {
         Classifier classifier = learner.train(train);
         //Make sure we can decode 2 labels
         assertNotNull(classifier.decodeLabel(0));
         assertNotNull(classifier.decodeLabel(1));
         assertNotNull(classifier.decodeLabel(2));
         assertNull(classifier.decodeLabel(3));

         assertNotNull(classifier.decodeFeature(1));
         assertTrue(classifier.encodeFeature("sepal_length") != -1);


         ClassifierEvaluation evaluation = new ClassifierEvaluation();
         evaluation.evaluate(classifier, test);
         assertEquals(targetAcc, evaluation.accuracy(), tolerance);
         System.out.println(learner.getClass().getSimpleName() + ": " + evaluation.accuracy());
         evaluation.output(System.out);
      });
   }


}//END OF ClassificationTest
