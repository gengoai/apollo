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

package com.gengoai.apollo.ml.classification;

import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Target2Label;
import cc.mallet.types.Alphabet;
import cc.mallet.types.InstanceList;
import cc.mallet.types.Labeling;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.params.ParamMap;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.ml.vectorizer.MalletVectorizer;
import com.gengoai.apollo.ml.vectorizer.VectorToTokensPipe;
import com.gengoai.conversion.Cast;

import java.util.Arrays;

/**
 * <p>
 * A wrapper around Mallet classifiers to work in the Apollo Classifier framework. All Mallet classifiers use a
 * specialized {@link com.gengoai.apollo.ml.vectorizer.Vectorizer} for labels and features.
 * </p>
 *
 * @author David B. Bracewell
 */
public abstract class MalletClassifier extends Classifier {
   private static final long serialVersionUID = 1L;
   /**
    * The Model.
    */
   protected cc.mallet.classify.Classifier model;

   /**
    * Instantiates a new Mallet classifier.
    *
    * @param preprocessors the preprocessors
    */
   public MalletClassifier(Preprocessor... preprocessors) {
      super(DiscretePipeline.create(new MalletVectorizer(new Alphabet()))
                            .update(p -> {
                               p.featureVectorizer = new MalletVectorizer(new Alphabet());
                               p.preprocessorList.addAll(preprocessors);
                            }));
   }

   /**
    * Create pipe serial pipes.
    *
    * @return the serial pipes
    */
   protected SerialPipes createPipe() {
      return new SerialPipes(Arrays.asList(new Target2Label(),
                                           new VectorToTokensPipe(Cast.as(getPipeline().featureVectorizer))));
   }

   @Override
   protected void fitPreprocessed(Dataset preprocessed, ParamMap fitParameters) {
      Pipe pipe = createPipe();
      pipe.setDataAlphabet(Cast.<MalletVectorizer>as(getPipeline().featureVectorizer).getAlphabet());
      InstanceList trainingData = new InstanceList(pipe);
      preprocessed.forEach(
         example -> trainingData.addThruPipe(new cc.mallet.types.Instance(example, example.getLabel(), null, null)));

      ClassifierTrainer<?> trainer = getTrainer(fitParameters);
      model = trainer.train(trainingData);
      MalletVectorizer labelVectorizer = Cast.as(getPipeline().labelVectorizer);
      labelVectorizer.setAlphabet(model.getInstancePipe().getTargetAlphabet());
   }

   /**
    * Gets the Mallet Classifier Trainer.
    *
    * @param parameters the parameters
    * @return the trainer
    */
   protected abstract ClassifierTrainer<?> getTrainer(ParamMap parameters);

   @Override
   public Classification predict(Example example) {
      Labeling labeling = model.classify(model.getInstancePipe()
                                              .instanceFrom(new cc.mallet.types.Instance(example, "", null, null)))
                               .getLabeling();
      double[] result = new double[getNumberOfLabels()];
      for (int i = 0; i < getNumberOfLabels(); i++) {
         result[i] = labeling.value(i);
      }
      return new Classification(NDArrayFactory.rowVector(result), getPipeline().labelVectorizer);
   }


}// END OF MalletClassifier
