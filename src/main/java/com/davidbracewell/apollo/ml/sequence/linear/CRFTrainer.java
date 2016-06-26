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

package com.davidbracewell.apollo.ml.sequence.linear;

import com.davidbracewell.apollo.ml.Dataset;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.sequence.Sequence;
import com.davidbracewell.apollo.ml.sequence.SequenceLabeler;
import com.davidbracewell.apollo.ml.sequence.SequenceLabelerLearner;
import com.davidbracewell.apollo.ml.sequence.TransitionFeatures;
import com.davidbracewell.io.Resources;
import third_party.org.chokkan.crfsuite.Attribute;
import third_party.org.chokkan.crfsuite.Item;
import third_party.org.chokkan.crfsuite.ItemSequence;
import third_party.org.chokkan.crfsuite.StringList;
import third_party.org.chokkan.crfsuite.Trainer;

/**
 * The type Crf trainer.
 *
 * @author David B. Bracewell
 */
public class CRFTrainer extends SequenceLabelerLearner {
  private static final long serialVersionUID = 1L;
  private Solver solver = Solver.LBFGS;
  private int maxIterations = 100;
  private double c2 = 1.0;

  @Override
  protected SequenceLabeler trainImpl(Dataset<Sequence> dataset) {
    LibraryLoader.INSTANCE.load();
    Trainer trainer = new Trainer();
    dataset.forEach(sequence -> {
      ItemSequence seq = new ItemSequence();
      StringList labels = new StringList();
      for (Instance instance : sequence.asInstances()) {
        Item item = new Item();
        instance.forEach(f -> item.add(new Attribute(f.getName(), f.getValue())));
        labels.add(instance.getLabel().toString());
        seq.add(item);
      }
      trainer.append(seq, labels, 0);
    });
    trainer.select(solver.parameterSetting, "crf1d");
    trainer.set("max_iterations", Integer.toString(maxIterations));
    trainer.set("c2", Double.toString(c2));


    String tempFile = Resources.temporaryFile().asFile().get().getAbsolutePath();
    trainer.train(tempFile, -1);
    return new CRFTagger(
      dataset.getLabelEncoder(),
      dataset.getFeatureEncoder(),
      dataset.getPreprocessors(),
      TransitionFeatures.FIRST_ORDER,
      tempFile,
      getValidator()
    );
  }

  @Override
  public void reset() {

  }

  /**
   * Gets solver.
   *
   * @return the solver
   */
  public Solver getSolver() {
    return solver;
  }

  /**
   * Sets solver.
   *
   * @param solver the solver
   */
  public void setSolver(Solver solver) {
    this.solver = solver;
  }

  /**
   * Gets max iterations.
   *
   * @return the max iterations
   */
  public int getMaxIterations() {
    return maxIterations;
  }

  /**
   * Sets max iterations.
   *
   * @param maxIterations the max iterations
   */
  public void setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
  }

  /**
   * Gets c 2.
   *
   * @return the c 2
   */
  public double getC2() {
    return c2;
  }

  /**
   * Sets c 2.
   *
   * @param c2 the c 2
   */
  public void setC2(double c2) {
    this.c2 = c2;
  }
}// END OF CRFTrainer
