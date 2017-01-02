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

package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.data.Dataset;
import org.junit.Test;

import java.util.Arrays;

import static com.davidbracewell.tuple.Tuples.$;
import static org.junit.Assert.*;

public abstract class BaseSequenceLabelerTest {

   private final SequenceLabelerLearner learner;
   private final double expectedAcc;
   private final double tolerance;

   protected BaseSequenceLabelerTest(SequenceLabelerLearner learner, double expectedAcc, double tolerance) {
      this.learner = learner;
      this.expectedAcc = expectedAcc;
      this.tolerance = tolerance;
   }

   Dataset<Sequence> getData() {
      return Dataset.sequence().source(
         Arrays.asList(Sequence.create($("The", "O"), $("black", "color"), $("train", "O"), $(".", "O")),
                       Sequence.create($("The", "O"), $("black", "color"), $("car", "O"), $(".", "O")),
                       Sequence.create($("The", "O"), $("red", "color"), $("bus", "O"), $(".", "O")),
                       Sequence.create($("The", "O"), $("red", "color"), $("boat", "O"), $(".", "O"))
                      ));
   }

   public void setup() {

   }

   @Test
   public void test() throws Exception {
      setup();
      SequenceLabeler labeler = learner.train(getData());
      PerInstanceEvaluation evaluation = new PerInstanceEvaluation();
      evaluation.evaluate(labeler, getData());
      assertEquals(expectedAcc, evaluation.eval.accuracy(), tolerance);
   }
}