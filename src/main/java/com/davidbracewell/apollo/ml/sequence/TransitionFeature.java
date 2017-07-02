package com.davidbracewell.apollo.ml.sequence;

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.function.SerializableFunction;
import com.davidbracewell.string.StringUtils;
import com.google.common.collect.Iterators;
import lombok.NonNull;

import java.io.Serializable;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * @author David B. Bracewell
 */
public interface TransitionFeature extends Serializable {

   TransitionFeature FIRST_ORDER = new TransitionFeature() {
      private static final long serialVersionUID = 1L;

      @Override
      public Iterator<String> extract(Context<Instance> iterator) {
         return Collections.singleton("T[0]=" + iterator.getLabel()).iterator();
      }
   };

   TransitionFeature SECOND_ORDER = new TransitionFeature() {
      private static final long serialVersionUID = 1L;

      @Override
      public Iterator<String> extract(Context<Instance> iterator) {
         String pLabel = "T[-2]=" + iterator.getPreviousLabel(-2).orElse("****START****");
         String cLabel = "T[-1]=" + iterator.getPreviousLabel(-1).orElse("****START****");
         return Arrays.asList(pLabel, pLabel + "::" + cLabel).iterator();
      }
   };

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
            String n = ii.getFeatureSpace()
                         .filter(f -> f.startsWith(name + "[0]="))
                         .findFirst()
                         .orElse(null);
            return n == null ? null : n.replace("[0]=", indexString);
         }
      };
   }

   public static void main(String[] args) {
      SequenceInput<String> si = new SequenceInput<>(Arrays.asList(
         "The", "dog", "on", "the", "hill"
                                                                  ));

      Sequence sequence = new SequenceFeaturizer<String>() {

         @Override
         public Set<Feature> apply(Context<String> stringContext) {
            return Collections.singleton(
               Feature.TRUE("Word[0]=" + stringContext.getCurrent()));
         }
      }.extractSequence(si.iterator());
      List<SerializableFunction<Context<Instance>, String>> l = parse("Word[-1],Word[0],Word[+1]");

      Context<Instance> cntx = sequence.iterator();
      while (cntx.hasNext()) {
         cntx.next();
         String featureName = l.stream()
                               .map(fun -> fun.apply(cntx))
                               .collect(Collectors.joining("::"));
         System.out.println(featureName + "\n");
      }

   }

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

   Iterator<String> extract(final Context<Instance> iterator);


}//END OF TransitionFeature
