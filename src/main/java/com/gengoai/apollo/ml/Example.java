package com.gengoai.apollo.ml;

import com.gengoai.Copyable;
import com.gengoai.collection.Streams;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;
import java.util.Set;
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
      return Streams.asStream(this)
                    .flatMap(e -> e.getFeatures().stream())
                    .map(f -> f.name)
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
      throw new UnsupportedOperationException();
   }

   /**
    * Sets the label for this example.
    *
    * @param label the new label
    * @throws UnsupportedOperationException If the example does not allow direct access to the label
    */
   public void setLabel(Object label) {
      throw new UnsupportedOperationException();
   }

   /**
    * Gets the label of this example as a double value
    *
    * @return the label as a double
    * @throws UnsupportedOperationException If the example does not allow direct access to the label
    */
   public double getLabelAsDouble() {
      return getLabel();
   }

   /**
    * Gets the label as a Set of string for multi-label problems
    *
    * @return the labels as a set of string
    * @throws UnsupportedOperationException If the example does not allow direct access to the label
    */
   public Set<String> getLabelAsSet() {
      throw new UnsupportedOperationException();
   }

   /**
    * Gets the label as a String value
    *
    * @return the label as a string value
    * @throws UnsupportedOperationException If the example does not allow direct access to the label
    */
   public String getLabelAsString() {
      return getLabel();
   }

   /**
    * Gets the label space of the example where the labels are strings. The label space is the set of distinct labels in
    * the example. This method will work for both singe and multi-examples where it will return a stream of labels
    * across all examples contained in this one for multi-examples.
    *
    * @return the label space
    */
   public Stream<String> getStringLabelSpace() {
      if (isMultiLabel()) {
         return Streams.asStream(this)
                       .flatMap(e -> e.getLabelAsSet().stream())
                       .distinct();

      }
      return Streams.asStream(this)
                    .map(Example::getLabelAsString)
                    .distinct();
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
      return false;
   }

   /**
    * Checks if this example is composed of multiple examples
    *
    * @return True if the example is composed of multiple examples, False otherwise
    */
   public boolean isMultiExample() {
      return !isSingleExample();
   }

   /**
    * Checks if the example's label has multiple values or not
    *
    * @return True if the example represents a multi-label example, False otherwise
    */
   public boolean isMultiLabel() {
      if (hasLabel()) {
         return getLabel() instanceof Collection;
      }
      return false;
   }

   /**
    * Checks if this example is a single example or not, which means that labels and features are retrievable.
    *
    * @return True if this is a single example, False if not.
    */
   public abstract boolean isSingleExample();

   @Override
   public ContextualIterator iterator() {
      return new ContextualIterator(this);
   }

   /**
    * The number of examples represented. For multi-example examples this is the number of sub-examples and for non
    * multi-example examples this is always 1.
    *
    * @return the number of  examples represented..
    */
   public abstract int size();

}//END OF Example
