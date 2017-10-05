package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.Interner;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.json.JsonReader;
import com.davidbracewell.json.JsonWriter;
import com.davidbracewell.tuple.Tuple;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * <p>A sequence is represented as an iterable of instances where each instance represents a single step or token.</p>
 *
 * @author David B. Bracewell
 */
public class Sequence implements Example, Serializable, Iterable<Instance> {
   /**
    * Signifies the Beginning of the sequence
    */
   public static final String BOS = "****START****";
   /**
    * Signifies the End of the sequence
    */
   public static final String EOS = "****END****";

   private static final long serialVersionUID = 1L;
   private final ArrayList<Instance> sequence;

   /**
    * Instantiates a new sequence from a list of instances.
    *
    * @param sequence the sequence
    */
   public Sequence(@NonNull List<Instance> sequence) {
      this.sequence = new ArrayList<>(sequence);
      this.sequence.trimToSize();
   }

   /**
    * Instantiates a new sequence from a variable length array of instances.
    *
    * @param instances the instances making up the sequences
    */
   public Sequence(@NonNull Instance... instances) {
      this.sequence = new ArrayList<>(Arrays.asList(instances));
      this.sequence.trimToSize();
   }

   /**
    * Instantiates a new empty sequence.
    */
   public Sequence() {
      this.sequence = new ArrayList<>();
   }

   /**
    * Creates a new sequence from variable length array of strings representing the words (instances) in the sequence.
    * Each word becomes a binary predicate set to true.
    *
    * @param words the words (instances) in the sequence.
    * @return the sequence
    */
   public static Sequence create(@NonNull String... words) {
      return create(Stream.of(words));
   }

   /**
    * Creates a new sequence from stream of strings representing the words (instances) in the sequence. Each word
    * becomes a binary predicate set to true.
    *
    * @param words the words (instances) in the sequence.
    * @return the sequence
    */
   public static Sequence create(@NonNull Stream<String> words) {
      return new Sequence(words.map(w -> Instance.create(Collections.singleton(Feature.TRUE(w))))
                               .collect(Collectors.toList()));
   }

   /**
    * Creates a new sequence from stream of tuples representing the words and their labels in the sequence. Each word
    * becomes a binary predicate set to true.
    *
    * @param labeledWords the words (instances) in the sequence.
    * @return the sequence
    */
   public static Sequence create(@NonNull Tuple... labeledWords) {
      return new Sequence(Stream.of(labeledWords)
                                .map(tuple -> {
                                   String name = tuple.get(0);
                                   if (tuple.degree() == 1) {
                                      return Instance.create(Collections.singleton(Feature.TRUE(name)));
                                   }
                                   return Instance.create(Collections.singleton(Feature.TRUE(name)), tuple.get(1));
                                })
                                .collect(Collectors.toList()));
   }

   public Instance asInstance(int labelIndex) {
      if (sequence.size() == 0) {
         return new Instance();
      }
      List<Feature> instFeatures = new ArrayList<>();
      int index = 0;
      for (Instance instance : sequence) {
         for (Feature feature : instance) {
            instFeatures.add(Feature.real(feature.getFeatureName() + "-" + index, feature.getValue()));
         }
         index++;
      }
      return Instance.create(instFeatures, sequence.get(labelIndex).getLabel());
   }

   @Override
   public List<Instance> asInstances() {
      return sequence;
   }

   @Override
   public Sequence copy() {
      return new Sequence(sequence.stream().map(Instance::copy).collect(Collectors.toList()));
   }

   @Override
   public void fromJson(JsonReader reader) throws IOException {
      sequence.clear();
      sequence.addAll(reader.nextCollection(ArrayList::new, "sequence", Instance.class));
      sequence.trimToSize();
   }

   /**
    * Get the instance at the given index.
    *
    * @param index the index of the instance to retrieve
    * @return the instance
    * @throws IndexOutOfBoundsException if the index is not in the sequence
    */
   public Instance get(int index) {
      return sequence.get(index);
   }

   @Override
   public Stream<String> getFeatureSpace() {
      return sequence.stream().flatMap(Instance::getFeatureSpace).distinct();
   }

   @Override
   public Stream<Object> getLabelSpace() {
      return sequence.stream().flatMap(Instance::getLabelSpace).distinct();
   }

   @Override
   public Sequence intern(Interner<String> interner) {
      return new Sequence(sequence.stream().map(instance -> instance.intern(interner)).collect(Collectors.toList()));
   }

   @Override
   public Context<Instance> iterator() {
      return new SequenceIterator();
   }

   /**
    * The number of instances in the sequence
    *
    * @return the size, in terms of instances, of the sequence.
    */
   public int size() {
      return sequence.size();
   }

   @Override
   public void toJson(JsonWriter writer) throws IOException {
      writer.property("sequence", sequence);
   }

   @Override
   public String toString() {
      return sequence.toString();
   }

   private class SequenceIterator extends Context<Instance> {
      private static final long serialVersionUID = 1L;

      @Override
      protected Optional<Instance> getContextAt(int index) {
         if (index < size() && index >= 0) {
            return Optional.of(get(index));
         }
         return Optional.empty();
      }

      @Override
      protected Optional<String> getLabelAt(int index) {
         if (index < size() && index >= 0) {
            return Optional.of(get(index).getLabel()).map(Object::toString);
         }
         return Optional.empty();
      }

      @Override
      public int size() {
         return Sequence.this.size();
      }
   }
}// END OF Sequence
