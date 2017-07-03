package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.function.SerializableFunction;
import com.davidbracewell.guava.common.collect.Iterators;
import com.davidbracewell.guava.common.collect.Lists;
import com.davidbracewell.string.StringUtils;
import lombok.NonNull;

import java.io.Serializable;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The interface Transition feature.
 *
 * @author David B. Bracewell
 */
public interface TransitionFeature extends Serializable {

   /**
    * The constant FIRST_ORDER.
    */
   TransitionFeature FIRST_ORDER = new TransitionFeature() {
      private static final long serialVersionUID = 1L;

      @Override
      public Iterator<String> extract(Context<Instance> iterator) {
         return Collections.singleton("T[0]=" + iterator.getLabel()).iterator();
      }
   };
   /**
    * The constant NO_OPT.
    */
   TransitionFeature NO_OPT = new TransitionFeature() {
      private static final long serialVersionUID = 1L;

      @Override
      public Iterator<String> extract(Context<Instance> iterator) {
         return Iterators.emptyIterator();
      }
   };
   /**
    * The constant SECOND_ORDER.
    */
   TransitionFeature SECOND_ORDER = new TransitionFeature() {
      private static final long serialVersionUID = 1L;

      @Override
      public Iterator<String> extract(Context<Instance> iterator) {
         String pLabel = "T[-2]=" + iterator.getPreviousLabel(-2).orElse("****START****");
         String cLabel = "T[-1]=" + iterator.getPreviousLabel(-1).orElse("****START****");
         return Arrays.asList(pLabel, pLabel + "::" + cLabel).iterator();
      }
   };

   /**
    * Chain transition feature.
    *
    * @param transitionFeatures the transition features
    * @return the transition feature
    */
   static TransitionFeature chain(@NonNull TransitionFeature... transitionFeatures) {
      if (transitionFeatures.length == 1) {
         return transitionFeatures[0];
      }
      return new TransitionFeature() {
         private static final long serialVersionUID = 1L;

         @Override
         public Iterator<String> extract(Context<Instance> iterator) {
            Iterator<String> itr = transitionFeatures[0].extract(iterator);
            for (int i = 1; i < transitionFeatures.length; i++) {
               itr = Iterators.concat(itr, transitionFeatures[i].extract(iterator));
            }
            return itr;
         }
      };
   }

   /**
    * Feature serializable function.
    *
    * @param name  the name
    * @param index the index
    * @return the serializable function
    */
   static SerializableFunction<Context<Instance>, String> feature(final String name, final int index) {
      return new SerializableFunction<Context<Instance>, String>() {
         private static final long serialVersionUID = 1L;

         @Override
         public String apply(Context<Instance> context) {
            Instance ii = context.getContext(index).orElse(null);
            String indexString = "["
                                    + ((index <= 0) ? index : "+" + index)
                                    + "]=";
            if (ii == null && index < 0) {
               return name + indexString + "****START****";
            }
            if (ii == null && index > 0) {
               return name + indexString + "****END****";
            }
            if (ii == null) {
               return null;
            }
            String n = ii.getFeatureSpace()
                         .filter(f -> f.startsWith(name + "[0]="))
                         .findFirst()
                         .orElse(null);
            return n == null ? null : n.replace("[0]=", indexString);
         }
      };
   }


   static TransitionFeature fromTemplate(@NonNull String template) {
      final List<SerializableFunction<Context<Instance>, String>> functions = parse(template);
      return new TransitionFeature() {
         private static final long serialVersionUID = 1L;

         @Override
         public Iterator<String> extract(Context<Instance> iterator) {
            if (functions.stream().map(fun -> fun.apply(iterator))
                         .anyMatch(Objects::isNull)) {
               return Collections.emptyIterator();
            }
            return functions.stream().map(fun -> fun.apply(iterator)).iterator();
         }
      };
   }

   /**
    * Parse list.
    *
    * @param template the template
    * @return the list
    */
   static List<SerializableFunction<Context<Instance>, String>> parse(String template) {
      final Pattern extractor = Pattern.compile("^(.*)\\[([\\-\\+]?\\d+)\\]$");
      List<String> elements = StringUtils.split(template, ',');
      List<SerializableFunction<Context<Instance>, String>> parts = new ArrayList<>();
      elements.forEach(p -> {
         Matcher m = extractor.matcher(p);
         m.find();
         int index = Integer.parseInt(m.group(2));
         if (m.group(1).equalsIgnoreCase("T")) {
            parts.add(unigramTag(index));
         } else {
            parts.add(feature(m.group(1), index));
         }
      });
      return parts;
   }

   /**
    * Unigram tag serializable function.
    *
    * @param index the index
    * @return the serializable function
    */
   static SerializableFunction<Context<Instance>, String> unigramTag(final int index) {
      return new SerializableFunction<Context<Instance>, String>() {
         private static final long serialVersionUID = 1L;

         @Override
         public String apply(Context<Instance> iterator) {
            return "T[" + index + "]=" + iterator.getPreviousLabel(-index)
                                                 .orElse("****START****");
         }
      };
   }

   /**
    * Extract iterator.
    *
    * @param iterator the iterator
    * @return the iterator
    */
   Iterator<String> extract(final Context<Instance> iterator);


   default void fit(@NonNull Dataset<Sequence> dataset) {
      dataset.getFeatureEncoder().fit(
         dataset.stream()
                .flatMap(sequence -> {
                            Context<Instance> ci = sequence.iterator();
                            Set<String> features = new HashSet<>();
                            while (ci.hasNext()) {
                               ci.next();
                               Iterator<String> itr = extract(ci);
                               while (itr.hasNext()) {
                                  features.add(itr.next());
                               }
                            }
                            return features.stream();
                         }
                        ));
   }

   default List<Instance> toInstances(Sequence sequence) {
      Context<Instance> itr = sequence.iterator();
      List<Instance> instances = new ArrayList<>();
      while (itr.hasNext()) {
         Instance instance = itr.next();
         List<Feature> features = Lists.newArrayList(instance);
         Iterator<String> transitions = extract(itr);
         while (transitions.hasNext()) {
            features.add(Feature.TRUE(transitions.next()));
         }
         instances.add(Instance.create(features, instance.getLabel()));
      }
      return instances;
   }

}//END OF TransitionFeature
