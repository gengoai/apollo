package com.gengoai.apollo.ml;

import com.gengoai.conversion.Cast;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import static com.gengoai.Validation.checkArgument;

/**
 * <p>A Sequence represents an example made up of a finite enumerated list of child examples ({@link Instance}s).
 * Sequences do not allow direct access to labels and features as the labels and features are associated with the
 * examples in the sequence.</p>
 *
 * @author David B. Bracewell
 */
public class Sequence extends Example {
   private static final long serialVersionUID = 1L;
   private final List<Instance> sequence = new ArrayList<>();

   /**
    * Instantiates a new Sequence with weight 1.0 with the given child examples.
    *
    * @param examples the child examples (i.e. sequence instances in order)
    */
   public Sequence(Example... examples) {
      this(Arrays.asList(examples));
   }

   /**
    * Instantiates a new Sequence with weight 1.0 with the given child examples.
    *
    * @param examples the child examples (i.e. sequence instances in order)
    */
   public Sequence(List<? extends Example> examples) {
      examples.forEach(this::add);
   }

   @Override
   public void add(Example example) {
      checkArgument(example instanceof Instance, "Can only add Instance examples.");
      sequence.add(Cast.as(example));
   }

   @Override
   public Example copy() {
      Sequence copy = new Sequence(this.sequence);
      copy.setWeight(getWeight());
      return copy;
   }

   @Override
   public boolean equals(Object o) {
      if (this == o) return true;
      if (!(o instanceof Sequence)) return false;
      Sequence examples = (Sequence) o;
      return Objects.equals(sequence, examples.sequence);
   }

   @Override
   public Example getExample(int index) {
      return sequence.get(index);
   }


   @Override
   public boolean isSingleExample() {
      return false;
   }

   @Override
   public int hashCode() {
      return Objects.hash(sequence);
   }

   @Override
   public int size() {
      return sequence.size();
   }

   @Override
   public String toString() {
      return "Sequence{" +
                "sequence=" + sequence +
                ", weight=" + getWeight() +
                '}';
   }
}//END OF Sequence
