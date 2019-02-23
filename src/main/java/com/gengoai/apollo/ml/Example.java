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

package com.gengoai.apollo.ml;

import com.gengoai.Copyable;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.collection.Streams;
import com.gengoai.conversion.Cast;
import com.gengoai.conversion.Converter;
import com.gengoai.conversion.TypeConversionException;
import com.gengoai.reflection.Types;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Stream;

import static com.gengoai.apollo.ml.Feature.booleanFeature;

/**
 * <p>Generic interface for representing a label and set of features. Classification and Regression problems use the
 * <code>Instance</code> specialization and Sequence Labeling Problems use the <code>Sequence</code>
 * specialization.</p>
 *
 * <p>Examples can be <code>Single</code> or <code>Multi</code> examples. Single examples, like {@link Instance}s,
 * represent a single observation and label and allow for labels and features to be retrieved. Multi-examples, like
 * {@link Sequence}s are containers that are made up of one or more other examples and do not themselves have labels or
 * features associated.</p>
 */
public abstract class Example implements Copyable<Example>, Iterable<Example>, Serializable {
   private static final long serialVersionUID = 1L;
   private double weight = 1.0;
   private Object label = null;

   /**
    * Adds an example. Will throw an <code>UnsupportedOperationException</code> if this is a not a multi-example.
    *
    * @param example the example to add
    * @throws UnsupportedOperationException if this is a not a multi-example.
    */
   public void add(Example example) {
      throw new UnsupportedOperationException();
   }

   /**
    * Gets the label as a String value
    *
    * @return the label as a string value
    * @throws UnsupportedOperationException If the example does not allow direct access to the label
    */
   public String getDiscreteLabel() {
      return getLabel();
   }

   /**
    * Gets the example at the given index.
    *
    * @param index the index of the example to get
    * @return the example at the given index.
    */
   public abstract Example getExample(int index);

   /**
    * Gets a feature from this example starting with the given prefix. If a feature is not found, a default true-valued
    * feature of <code>prefix+"UNKNOWN</code> is returned. (Note that child classes may override this to return
    * different default features.
    *
    * @param prefix the prefix to search for
    * @return the first feature whose name starts with the given prefix or a new feature with the given prefix and
    * UNKNOWN.
    */
   public Feature getFeatureByPrefix(String prefix) {
      return getFeatureByPrefix(prefix, booleanFeature(prefix + "UNKNOWN"));
   }

   /**
    * Gets a feature from this example starting with the given prefix. If a feature is not found, the given default is
    * returned.
    *
    * @param prefix       the prefix to search for
    * @param defaultValue the default feature to return if one is not found
    * @return the first feature whose name starts with the given prefix or the default value
    */
   public Feature getFeatureByPrefix(String prefix, Feature defaultValue) {
      throw new UnsupportedOperationException();
   }

   /**
    * Gets the feature space of the example. The feature space is the set of distinct feature names in the example. This
    * method will work for both singe and multi-examples where it will return a stream of names across all examples
    * contained in this one for multi-examples.
    *
    * @return the feature space
    */
   public Stream<String> getFeatureNameSpace() {
      return stream().flatMap(e -> e.getFeatures().stream())
                     .map(Feature::getName)
                     .distinct();
   }

   /**
    * Gets the features associated with this example
    *
    * @return the list of features associated with this example
    * @throws UnsupportedOperationException If the example does not allow direct access to the features
    */
   public List<Feature> getFeatures() {
      throw new UnsupportedOperationException();
   }

   /**
    * Gets the label associated with the example.
    *
    * @param <T> the label type parameter
    * @return the label
    * @throws UnsupportedOperationException If the example does not allow direct access to the label
    */
   public <T> T getLabel() {
      return Cast.as(label);
   }

   /**
    * Sets the label for this example.
    *
    * @param label the new label
    * @return the label
    * @throws UnsupportedOperationException If the example does not allow direct access to the label
    */
   public Example setLabel(Object label) {
      if (label == null) {
         this.label = null;
      } else if (label instanceof Number) {
         this.label = Cast.<Number>as(label).doubleValue();
      } else if (label instanceof CharSequence) {
         this.label = label.toString();
      } else if (label instanceof Iterator || label instanceof Iterable ||
                    label instanceof Stream || label.getClass().isArray()
      ) {
         try {
            this.label = Converter.convert(label, Types.parameterizedType(Set.class, String.class));
         } catch (TypeConversionException e) {
            throw new IllegalArgumentException("Unable to set (" + label + ") as the Instance's label");
         }
      } else {
         throw new IllegalArgumentException("Unable to set (" + label + ") as the Instance's label");
      }
      return this;
   }

   /**
    * Gets the label space of the example where the labels are strings. The label space is the set of distinct labels in
    * the example. This method will work for both singe and multi-examples where it will return a stream of labels
    * across all examples contained in this one for multi-examples.
    *
    * @return the label space
    */
   public Stream<String> getLabelSpace() {
      if (hasLabel()) {
         return getMultiLabel().stream();
      }
      return stream().flatMap(e -> e.getMultiLabel().stream()).distinct();
   }

   /**
    * Gets the label as a Set of string for multi-label problems
    *
    * @return the labels as a set of string
    * @throws UnsupportedOperationException If the example does not allow direct access to the label
    */
   public Set<String> getMultiLabel() {
      Object lbl = getLabel();
      if (lbl == null) {
         return null;
      }
      if (lbl instanceof Set) {
         return Cast.as(lbl);
      }
      return Collections.singleton(lbl.toString());
   }

   /**
    * Gets the label of this example as a double value
    *
    * @return the label as a double
    * @throws UnsupportedOperationException If the example does not allow direct access to the label
    */
   public double getNumericLabel() {
      if (getLabel() instanceof CharSequence) {
         return Double.parseDouble(getDiscreteLabel());
      }
      return getLabel();
   }

   /**
    * Gets the weight of the example
    *
    * @return the weight
    */
   public final double getWeight() {
      return weight;
   }

   /**
    * Sets the weight of the example
    *
    * @param weight the weight
    */
   public final void setWeight(double weight) {
      this.weight = weight;
   }

   /**
    * Checks if the example has a label assigned to it or not.
    *
    * @return True if a label is assigned, False otherwise.
    */
   public boolean hasLabel() {
      return label != null;
   }

   /**
    * Checks if this an instance (or leaf level) example that will contain features and labels.
    *
    * @return True if an instance, false otherwise
    */
   public boolean isInstance() {
      return false;
   }


   @Override
   public ContextualIterator iterator() {
      return new ContextualIterator(this);
   }

   /**
    * Generates a new example from this one passing the features through the given mapper. The mapper returns an
    * <code>Optional</code> for cases where the feature should be dropped. The resulting example will have the same
    * weight and label.
    *
    * @param mapper the mapper
    * @return the modified example
    */
   public Example mapFeatures(Function<? super Feature, Optional<Feature>> mapper) {
      throw new UnsupportedOperationException();
   }

   /**
    * Maps each of the instances of this example.
    *
    * @param mapper the mapper
    * @return the example
    */
   public abstract Example mapInstance(Function<Instance, Instance> mapper);

   /**
    * Preprocesses the example using the given {@link Pipeline} and then transforms the result into an {@link NDArray}.
    *
    * @param pipeline the pipeline to use for preprocessing and vectorization
    * @return the NDArray
    */
   public final NDArray preprocessAndTransform(Pipeline pipeline) {
      return pipeline.preprocessorList.apply(this).transform(pipeline);
   }

   /**
    * The number of examples represented. For multi-example examples this is the number of sub-examples and for non
    * multi-example examples this is always 1.
    *
    * @return the number of  examples represented..
    */
   public abstract int size();

   /**
    * Creates a stream across the examples in this example
    *
    * @return the stream of examples
    */
   public Stream<Example> stream() {
      return Streams.asStream(this);
   }

   /**
    * Transforms this example into an {@link NDArray} using the vectorizers of the given {@link Pipeline}
    *
    * @param pipeline the pipeline to use for vectorization
    * @return the NDArray
    */
   public abstract NDArray transform(Pipeline pipeline);


}//END OF Example
