package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.Interner;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.io.structured.StructuredReader;
import com.davidbracewell.io.structured.StructuredWriter;
import lombok.NonNull;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * The type Sequence.
 *
 * @author David B. Bracewell
 */
public class Sequence implements Example, Serializable, Iterable<Instance> {

   public static final String BOS = "****START****";
   public static final String EOS = "****END****";

   private static final long serialVersionUID = 1L;
   private final ArrayList<Instance> sequence;

   /**
    * Instantiates a new Sequence.
    *
    * @param sequence the sequence
    */
   public Sequence(@NonNull List<Instance> sequence) {
      this.sequence = new ArrayList<>(sequence);
      this.sequence.trimToSize();
   }

   public Sequence(@NonNull Instance... words) {
      this.sequence = new ArrayList<>(Arrays.asList(words));
      this.sequence.trimToSize();
   }

   public Sequence() {
      this.sequence = new ArrayList<>();
   }

   public static Sequence toSequence(String... words) {
      return toSequence(Stream.of(words));
   }

   public static Sequence toSequence(Stream<String> words) {
      return new Sequence(words.map(w -> Instance.create(Collections.singleton(Feature.TRUE(w))))
                               .collect(Collectors.toList()));
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
      return new Sequence(
                            sequence.stream().map(instance -> instance.intern(interner)).collect(Collectors.toList())
      );
   }

   @Override
   public Sequence copy() {
      return new Sequence(
                            sequence.stream().map(Instance::copy).collect(Collectors.toList())
      );
   }

   /**
    * Get instance.
    *
    * @param index the index
    * @return the instance
    */
   public Instance get(int index) {
      return sequence.get(index);
   }

   /**
    * Size int.
    *
    * @return the int
    */
   public int size() {
      return sequence.size();
   }

   /**
    * Iterator sequence iterator.
    *
    * @return the sequence iterator
    */
   public Context<Instance> iterator() {
      return new SequenceIterator();
   }

   @Override
   public List<Instance> asInstances() {
      return sequence;
   }

   @Override
   public String toString() {
      return sequence.toString();
   }

   @Override
   public void read(StructuredReader reader) throws IOException {
      sequence.clear();
      sequence.addAll(reader.nextCollection(ArrayList::new, "sequence", Instance.class));
      sequence.trimToSize();
   }

   @Override
   public void write(StructuredWriter writer) throws IOException {
      writer.writeKeyValue("sequence", sequence);
   }

   private class SequenceIterator extends Context<Instance> {
      private static final long serialVersionUID = 1L;

      @Override
      public int size() {
         return Sequence.this.size();
      }

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
   }
}// END OF Sequence
