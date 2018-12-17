package com.gengoai.apollo.ml;

import com.gengoai.Validation;
import com.gengoai.conversion.Cast;
import com.gengoai.conversion.Converter;
import com.gengoai.conversion.TypeConversionException;
import com.gengoai.reflection.Types;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static com.gengoai.apollo.ml.Feature.booleanFeature;

/**
 * <p>An Instance represents an example over a single object. It has a {@link #size()} of <code>1</code> and is not
 * allowed to add any child examples. Instances are used for input to classification and regression problems and as the
 * child examples for sequence labeling problems.</p>
 *
 * @author David B. Bracewell
 */
public class Instance extends Example {
   private static final long serialVersionUID = 1L;
   private final List<Feature> features;
   private Object label;


   /**
    * Instantiates a new Instance with a null label and no features defined.
    */
   public Instance() {
      this.features = new ArrayList<>();
      this.label = null;
   }

   /**
    * Instantiates a new Instance with the given label and features.
    *
    * @param label    the label
    * @param features the features
    */
   public Instance(Object label, Feature... features) {
      this(label, Arrays.asList(features));
   }

   /**
    * Instantiates a new Instance with  the given label and features.
    *
    * @param label    the label
    * @param features the features
    */
   public Instance(Object label, List<Feature> features) {
      this.features = new ArrayList<>(features);
      setLabel(label);
   }

   /**
    * Creates a special Instance denoting the begin of sequence (or sentence), which has a label and single True feature
    * named <code>__BOS-INDEX__</code>, where <code>INDEX</code> is the offset from the <code>0</code> index. This
    * instance will return the single feature name with given prefix on all calls to {@link
    * #getFeatureByPrefix(String)}.
    *
    * @param distanceFromBegin the offset from the beginning of the sequence (i.e. index 0, e.g. -1)
    * @return the special beginning of sequence example at the given offset
    */
   public static Example BEGIN_OF_SEQUENCE(int distanceFromBegin) {
      String name = "__BOS-" + Math.abs(distanceFromBegin) + "__";
      return new Instance(name, booleanFeature(name)) {
         @Override
         public Feature getFeatureByPrefix(String prefix) {
            return booleanFeature(prefix + name);
         }
      };
   }

   /**
    * Creates a special Instance denoting the end of sequence (or sentence), which has a label and single True feature
    * named <code>__EOS-INDEX+1__</code>, where <code>INDEX</code> is the offset from the size of the example in the
    * sequence. This instance will return the single feature name with given prefix on all calls to {@link
    * #getFeatureByPrefix(String)}.
    *
    * @param distanceFromEnd the offset from the size of the example. (e.g. if the size is 4 an offset could be 0 when
    *                        the index is at 4, 1 when the index is at 5, etc.)
    * @return the special end of sequence example at the given offset
    */
   public static Example END_OF_SEQUENCE(int distanceFromEnd) {
      String name = "__EOS-" + (distanceFromEnd + 1) + "__";
      return new Instance(name, booleanFeature(name)) {
         @Override
         public Feature getFeatureByPrefix(String prefix) {
            return booleanFeature(prefix + name);
         }
      };
   }

   @Override
   public Example copy() {
      Instance copy = new Instance(label, features);
      copy.setWeight(getWeight());
      return copy;
   }

   @Override
   public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Instance)) return false;
      Instance instance = (Instance) o;
      return Objects.equals(features, instance.features) &&
                Objects.equals(label, instance.label);
   }

   @Override
   public Example getExample(int index) {
      Validation.checkPositionIndex(index, 1);
      return this;
   }

   @Override
   public Feature getFeatureByPrefix(String prefix, Feature defaultValue) {
      return features.stream()
                     .filter(f -> f.hasPrefix(prefix))
                     .findFirst()
                     .orElse(defaultValue);
   }

   @Override
   public List<Feature> getFeatures() {
      return features;
   }

   @Override
   public <T> T getLabel() {
      return Cast.as(label);
   }

   @Override
   public void setLabel(Object label) {
      if (label == null) {
         this.label = null;
      } else if (label instanceof Number) {
         this.label = Cast.<Number>as(label).doubleValue();
      } else if (label.getClass().isArray() && label.getClass().getComponentType().isPrimitive()) {
         try {
            this.label = Converter.convert(label, float[].class);
         } catch (TypeConversionException e) {
            throw new IllegalArgumentException("Unable to set (" + label + ") as the Instance's label");
         }
      } else if (label instanceof CharSequence) {
         this.label = label.toString();
      } else if (label instanceof Iterator
                    || label instanceof Iterable
                    || label instanceof Stream
                    || label.getClass().isArray()
      ) {
         try {
            this.label = Converter.convert(label, Types.parameterizedType(Set.class, String.class));
         } catch (TypeConversionException e) {
            throw new IllegalArgumentException("Unable to set (" + label + ") as the Instance's label");
         }
      } else {
         throw new IllegalArgumentException("Unable to set (" + label + ") as the Instance's label");
      }
   }

   @Override
   public boolean isSingleExample() {
      return true;
   }

   @Override
   public int hashCode() {
      return Objects.hash(features, label);
   }

   @Override
   public int size() {
      return 1;
   }

   @Override
   public Example mapFeatures(Function<? super Feature, ? extends Feature> mapper) {
      return new Instance(label, features.stream().map(mapper).filter(Objects::nonNull).collect(Collectors.toList()));
   }

   @Override
   public Example mapLabel(Function<? super Object, ? extends Object> mapper) {
      Example e = copy();
      e.setLabel(mapper.apply(label));
      return e;
   }

   @Override
   public String toString() {
      return "Instance{" +
                "features=" + features +
                ", label=" + label +
                ", weight=" + getWeight() +
                '}';
   }

   @Override
   public Set<String> getLabelAsSet() {
      Object lbl = getLabel();
      if (lbl instanceof Set) {
         return Cast.as(lbl);
      }
      return Collections.singleton(lbl.toString());
   }

   @Override
   public boolean hasLabel() {
      return label != null;
   }
}//END OF Instance
