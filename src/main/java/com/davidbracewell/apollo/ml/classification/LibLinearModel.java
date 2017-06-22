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

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.collection.counter.HashMapMultiCounter;
import com.davidbracewell.collection.counter.MultiCounter;
import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;

import java.util.Iterator;

/**
 * A model trained using LibLinear
 *
 * @author David B. Bracewell
 */
public class LibLinearModel extends Classifier {
   private static final long serialVersionUID = 1L;
   protected Model model;
   protected int biasIndex = 0;

   protected LibLinearModel(ClassifierLearner learner) {
      super(learner);
   }


   /**
    * Converts an Apollo vector into an array of LibLinear feature nodes
    *
    * @param vector    the vector to convert
    * @param biasIndex the index of the bias variable (<0 for no bias)
    * @return the feature node array
    */
   public static Feature[] toFeature(Vector vector, int biasIndex) {
      int size = vector.size() + (biasIndex > 0 ? 1 : 0);
      final Feature[] feature = new Feature[size];
      int index = 0;
      for (Iterator<Vector.Entry> itr = vector.orderedNonZeroIterator(); itr.hasNext(); index++) {
         Vector.Entry entry = itr.next();
         feature[index] = new FeatureNode(entry.index + 1, entry.value);
      }
      if (biasIndex > 0) {
         feature[size - 1] = new FeatureNode(biasIndex, 1.0);
      }
      return feature;
   }

   @Override
   public Classification classify(Vector vector) {
      double[] p = new double[numberOfLabels()];
      if (model.isProbabilityModel()) {
         Linear.predictProbability(model, toFeature(vector, biasIndex), p);
      } else {
         Linear.predictValues(model, toFeature(vector, biasIndex), p);
      }

      //re-arrange the probabilities to match the target feature
      double[] prime = new double[numberOfLabels()];
      int[] labels = model.getLabels();
      for (int i = 0; i < labels.length; i++) {
         prime[labels[i]] = p[i];
      }

      return createResult(prime);
   }

   @Override
   public MultiCounter<String, String> getModelParameters() {
      double[] modelWeights = model.getFeatureWeights();
      int[] labels = model.getLabels();
      MultiCounter<String, String> weights = new HashMapMultiCounter<>();

      int nrClass = model.getNrClass() - 1;
      for (int i = 0, index = 0; i < getFeatureEncoder().size(); i++, index += nrClass) {
         String featureName = getFeatureEncoder().decode(i).toString();
         for (int j = 0; j < nrClass; j++) {
            weights.set(featureName, getLabelEncoder().decode(labels[j]).toString(), modelWeights[j + index]);
         }
      }

      if (modelWeights.length > (nrClass * model.getNrFeature())) {
         for (int j = modelWeights.length - nrClass, ci = 0; j < modelWeights.length; j++, ci++) {
            weights.set("**BIAS**", getLabelEncoder().decode(labels[ci]).toString(), modelWeights[j]);
         }
      }

      return weights;
   }

}//END OF LibLinearModel
