## Training Datasets

<table style="border-collapse: collapse;">
  <thead>
    <tr>
      <th style="border-right: 1px solid #94a3b8;">Dataset</th>
      <th style="border-right: 1px solid #94a3b8;">Measures</th>
      <th style="text-align: center; border-right: 1px solid #94a3b8;">Abstention</th>
      <th style="text-align: center;">Fluency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>KUQ</b></td>
      <td style="border-right: 1px solid #94a3b8;">Answerable vs. unanswerable knowledge questions</td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>SQuAD</b></td>
      <td style="border-right: 1px solid #94a3b8;">Reading comprehension with unanswerable variants</td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>UltraChat</b></td>
      <td style="border-right: 1px solid #94a3b8;">Multi-turn chat completions; perplexity measures generation quality</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="text-align: center; color: #16a34a; font-size: 1.2em;">✓</td>
    </tr>
  </tbody>
</table>

---

## Evaluation Datasets

<table style="border-collapse: collapse;">
  <thead>
    <tr>
      <th style="border-right: 1px solid #94a3b8;">Dataset</th>
      <th style="border-right: 1px solid #94a3b8;">Measures</th>
      <th style="text-align: center; border-right: 1px solid #94a3b8;">Abstention</th>
      <th style="text-align: center; border-right: 1px solid #94a3b8;">Hallucination</th>
      <th style="text-align: center; border-right: 1px solid #94a3b8;">Misinformation</th>
      <th style="text-align: center;">Fluency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>KUQ</b></td>
      <td style="border-right: 1px solid #94a3b8;">Answerable vs. unanswerable knowledge questions</td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>SQuAD</b></td>
      <td style="border-right: 1px solid #94a3b8;">Reading comprehension with unanswerable variants</td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>SelfAware</b></td>
      <td style="border-right: 1px solid #94a3b8;">Intrinsic knowledge limits (no external context)</td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>FaithEval</b></td>
      <td style="border-right: 1px solid #94a3b8;">Context-grounded questions with no faithful answer</td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>NoMIRACL</b></td>
      <td style="border-right: 1px solid #94a3b8;">Retrieval QA where retrieved passages may not support an answer</td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>PopQA</b></td>
      <td style="border-right: 1px solid #94a3b8;">Open-domain factual QA with popularity-stratified entities</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>TruthfulQA</b></td>
      <td style="border-right: 1px solid #94a3b8;">Questions designed to elicit common misconceptions</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>SciFact</b></td>
      <td style="border-right: 1px solid #94a3b8;">Scientific claim verification (Supports / Refutes / Not Enough Info)</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>AVeritec</b></td>
      <td style="border-right: 1px solid #94a3b8;">Real-world claim verification with Conflicting evidence label</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="text-align: center; color: #16a34a; border-right: 1px solid #94a3b8; font-size: 1.2em;">✓</td>
      <td></td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>UltraChat</b></td>
      <td style="border-right: 1px solid #94a3b8;">Multi-turn chat completions; perplexity measures generation quality</td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="border-right: 1px solid #94a3b8;"></td>
      <td style="text-align: center; color: #16a34a; font-size: 1.2em;">✓</td>
    </tr>
  </tbody>
</table>

---

## UOC vs Zero-Shot Baseline — KUQ · SQuAD · UltraChat

<table style="border-collapse: collapse;">
  <thead>
    <tr>
      <th rowspan="2" style="border-right: 1px solid #94a3b8;">Model</th>
      <th rowspan="2" style="border-right: 1px solid #94a3b8;">Method</th>
      <th colspan="3" align="center" style="text-align: center; border-right: 1px solid #94a3b8;">KUQ</th>
      <th colspan="3" align="center" style="text-align: center; border-right: 1px solid #94a3b8;">SQuAD</th>
      <th align="center" style="text-align: center;">UltraChat</th>
    </tr>
    <tr>
      <th style="text-align: center;">Decision Accuracy ↑</th>
      <th style="text-align: center;">False Commit ↓</th>
      <th style="text-align: center; border-right: 1px solid #94a3b8;">False Abstain ↓</th>
      <th style="text-align: center;">Decision Accuracy ↑</th>
      <th style="text-align: center;">False Commit ↓</th>
      <th style="text-align: center; border-right: 1px solid #94a3b8;">False Abstain ↓</th>
      <th style="text-align: center;">PPL ratio ↓</th>
    </tr>
  </thead>
  <tbody>
    <!-- Qwen 3.5 9B -->
    <tr>
      <td rowspan="3" style="border-right: 1px solid #94a3b8;"><b>Qwen 3.5 9B</b></td>
      <td style="border-right: 1px solid #94a3b8;">Baseline</td>
      <td align="right">0.679</td>
      <td align="right">0.604</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.038</td>
      <td align="right">0.743</td>
      <td align="right">0.482</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.032</td>
      <td align="right">1.000</td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>UOC (Ours)</b></td>
      <td align="right"><b>0.941</b></td>
      <td align="right"><b>0.088</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;"><b>0.030</b></td>
      <td align="right"><b>0.960</b></td>
      <td align="right"><b>0.050</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;"><b>0.030</b></td>
      <td align="right">0.997</td>
    </tr>
    <tr style="background-color: #fef9c3;">
      <td style="color: #92400e; border-right: 1px solid #94a3b8;"><b>Δ</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.262</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.516</b></td>
      <td align="right" style="color: #16a34a; border-right: 1px solid #94a3b8;"><b>−0.008</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.217</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.432</b></td>
      <td align="right" style="color: #16a34a; border-right: 1px solid #94a3b8;"><b>−0.002</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.3%</b></td>
    </tr>
    <!-- GPT-OSS 20B -->
    <tr>
      <td rowspan="3" style="border-right: 1px solid #94a3b8;"><b>GPT-OSS 20B</b></td>
      <td style="border-right: 1px solid #94a3b8;">Baseline</td>
      <td align="right">0.546</td>
      <td align="right">0.858</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.050</td>
      <td align="right">0.595</td>
      <td align="right">0.804</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.006</td>
      <td align="right">1.000</td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>UOC (Ours)</b></td>
      <td align="right"><b>0.817</b></td>
      <td align="right"><b>0.327</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;"><b>0.040</b></td>
      <td align="right"><b>0.880</b></td>
      <td align="right"><b>0.200</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.040</td>
      <td align="right">1.021</td>
    </tr>
    <tr style="background-color: #fef9c3;">
      <td style="color: #92400e; border-right: 1px solid #94a3b8;"><b>Δ</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.271</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.531</b></td>
      <td align="right" style="color: #16a34a; border-right: 1px solid #94a3b8;"><b>−0.010</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.285</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.604</b></td>
      <td align="right" style="color: #dc2626; border-right: 1px solid #94a3b8;">+0.034</td>
      <td align="right" style="color: #dc2626;">+2.1%</td>
    </tr>
  </tbody>
</table>

> **False Commit** = answers when it should abstain &nbsp;·&nbsp; **False Abstain** = abstains when it should answer &nbsp;·&nbsp; **PPL ratio** = UOC PPL / Baseline PPL (1.00 = no change)

---

## UOC vs Zero-Shot Baseline — SelfAware · FaithEval · NoMIRACL

<table style="border-collapse: collapse;">
  <thead>
    <tr>
      <th rowspan="2" style="border-right: 1px solid #94a3b8;">Model</th>
      <th rowspan="2" style="border-right: 1px solid #94a3b8;">Method</th>
      <th colspan="3" align="center" style="text-align: center; border-right: 1px solid #94a3b8;">SelfAware</th>
      <th colspan="3" align="center" style="text-align: center; border-right: 1px solid #94a3b8;">FaithEval</th>
      <th colspan="3" align="center" style="text-align: center;">NoMIRACL</th>
    </tr>
    <tr>
      <th style="text-align: center;">Decision Accuracy ↑</th>
      <th style="text-align: center;">False Commit ↓</th>
      <th style="text-align: center; border-right: 1px solid #94a3b8;">False Abstain ↓</th>
      <th style="text-align: center;">Decision Accuracy ↑</th>
      <th style="text-align: center;">False Commit ↓</th>
      <th style="text-align: center; border-right: 1px solid #94a3b8;">False Abstain ↓</th>
      <th style="text-align: center;">Decision Accuracy ↑</th>
      <th style="text-align: center;">False Commit ↓</th>
      <th style="text-align: center;">False Abstain ↓</th>
    </tr>
  </thead>
  <tbody>
    <!-- Qwen 3.5 9B -->
    <tr>
      <td rowspan="3" style="border-right: 1px solid #94a3b8;"><b>Qwen 3.5 9B</b></td>
      <td style="border-right: 1px solid #94a3b8;">Baseline</td>
      <td align="right">0.623</td>
      <td align="right">0.734</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.020</td>
      <td align="right">0.671</td>
      <td align="right">0.329</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">—</td>
      <td align="right">0.786</td>
      <td align="right">0.390</td>
      <td align="right">0.038</td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>UOC (Ours)</b></td>
      <td align="right"><b>0.846</b></td>
      <td align="right"><b>0.180</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.128</td>
      <td align="right"><b>0.849</b></td>
      <td align="right"><b>0.151</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;">—</td>
      <td align="right"><b>0.883</b></td>
      <td align="right"><b>0.204</b></td>
      <td align="right"><b>0.030</b></td>
    </tr>
    <tr style="background-color: #fef9c3;">
      <td style="color: #92400e; border-right: 1px solid #94a3b8;"><b>Δ</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.223</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.554</b></td>
      <td align="right" style="color: #dc2626; border-right: 1px solid #94a3b8;">+0.108</td>
      <td align="right" style="color: #16a34a;"><b>+0.178</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.178</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;">—</td>
      <td align="right" style="color: #16a34a;"><b>+0.097</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.186</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.008</b></td>
    </tr>
    <!-- GPT-OSS 20B -->
    <tr>
      <td rowspan="3" style="border-right: 1px solid #94a3b8;"><b>GPT-OSS 20B</b></td>
      <td style="border-right: 1px solid #94a3b8;">Baseline</td>
      <td align="right">0.529</td>
      <td align="right">0.908</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.034</td>
      <td align="right">0.143</td>
      <td align="right">0.857</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">—</td>
      <td align="right">0.566</td>
      <td align="right">0.832</td>
      <td align="right">0.036</td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>UOC (Ours)</b></td>
      <td align="right"><b>0.689</b></td>
      <td align="right"><b>0.592</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;"><b>0.030</b></td>
      <td align="right"><b>0.664</b></td>
      <td align="right"><b>0.336</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;">—</td>
      <td align="right"><b>0.790</b></td>
      <td align="right"><b>0.380</b></td>
      <td align="right">0.040</td>
    </tr>
    <tr style="background-color: #fef9c3;">
      <td style="color: #92400e; border-right: 1px solid #94a3b8;"><b>Δ</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.160</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.316</b></td>
      <td align="right" style="color: #16a34a; border-right: 1px solid #94a3b8;"><b>−0.004</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.521</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.521</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;">—</td>
      <td align="right" style="color: #16a34a;"><b>+0.224</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.452</b></td>
      <td align="right" style="color: #dc2626;">+0.004</td>
    </tr>
  </tbody>
</table>

> † FaithEval contains only unanswerable instances — False Abstain is not applicable (—)

---

## UOC vs Zero-Shot Baseline — PopQA · TruthfulQA

<table style="border-collapse: collapse;">
  <thead>
    <tr>
      <th rowspan="2" style="border-right: 1px solid #94a3b8;">Model</th>
      <th rowspan="2" style="border-right: 1px solid #94a3b8;">Method</th>
      <th colspan="3" align="center" style="text-align: center; border-right: 1px solid #94a3b8;">PopQA</th>
      <th colspan="3" align="center" style="text-align: center;">TruthfulQA</th>
    </tr>
    <tr>
      <th style="text-align: center;">Hallucination Rate ↓</th>
      <th style="text-align: center;">Correct Rate ↑</th>
      <th style="text-align: center; border-right: 1px solid #94a3b8;">Abstention Rate ↑</th>
      <th style="text-align: center;">Hallucination Rate ↓</th>
      <th style="text-align: center;">Correct Rate ↑</th>
      <th style="text-align: center;">Abstention Rate ↑</th>
    </tr>
  </thead>
  <tbody>
    <!-- Qwen 3.5 9B -->
    <tr>
      <td rowspan="3" style="border-right: 1px solid #94a3b8;"><b>Qwen 3.5 9B</b></td>
      <td style="border-right: 1px solid #94a3b8;">Baseline</td>
      <td align="right">0.641</td>
      <td align="right">0.351</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.008</td>
      <td align="right">0.302</td>
      <td align="right">0.693</td>
      <td align="right">0.005</td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>UOC (Ours)</b></td>
      <td align="right"><b>0.113</b></td>
      <td align="right"><b>0.378</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;"><b>0.509</b></td>
      <td align="right"><b>0.206</b></td>
      <td align="right"><b>0.719</b></td>
      <td align="right"><b>0.076</b></td>
    </tr>
    <tr style="background-color: #fef9c3;">
      <td style="color: #92400e; border-right: 1px solid #94a3b8;"><b>Δ</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.528</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.027</b></td>
      <td align="right" style="color: #16a34a; border-right: 1px solid #94a3b8;"><b>+0.501</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.096</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.026</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.071</b></td>
    </tr>
    <!-- GPT-OSS 20B -->
    <tr>
      <td rowspan="3" style="border-right: 1px solid #94a3b8;"><b>GPT-OSS 20B</b></td>
      <td style="border-right: 1px solid #94a3b8;">Baseline</td>
      <td align="right">0.567</td>
      <td align="right">0.424</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.009</td>
      <td align="right">0.469</td>
      <td align="right">0.513</td>
      <td align="right">0.017</td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>UOC (Ours)</b></td>
      <td align="right"><b>0.443</b></td>
      <td align="right"><b>0.489</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;"><b>0.068</b></td>
      <td align="right"><b>0.348</b></td>
      <td align="right"><b>0.588</b></td>
      <td align="right"><b>0.065</b></td>
    </tr>
    <tr style="background-color: #fef9c3;">
      <td style="color: #92400e; border-right: 1px solid #94a3b8;"><b>Δ</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.124</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.065</b></td>
      <td align="right" style="color: #16a34a; border-right: 1px solid #94a3b8;"><b>+0.059</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.121</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.075</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.048</b></td>
    </tr>
  </tbody>
</table>

> **Hallucination Rate** = fraction of wrong committed answers &nbsp;·&nbsp; **Abstention Rate** = fraction where the model declined to answer

---

## UOC vs Zero-Shot Baseline — SciFact · AVeritec

<table style="border-collapse: collapse;">
  <thead>
    <tr>
      <th rowspan="2" style="border-right: 1px solid #94a3b8;">Model</th>
      <th rowspan="2" style="border-right: 1px solid #94a3b8;">Method</th>
      <th colspan="4" align="center" style="text-align: center; border-right: 1px solid #94a3b8;">SciFact</th>
      <th colspan="5" align="center" style="text-align: center;">AVeritec</th>
    </tr>
    <tr>
      <th style="text-align: center;">Overall Accuracy ↑</th>
      <th style="text-align: center;">Missed-NOINFO ↓</th>
      <th style="text-align: center;">Over-SUPPORTS ↓</th>
      <th style="text-align: center; border-right: 1px solid #94a3b8;">Over-REFUTES ↓</th>
      <th style="text-align: center;">Overall Accuracy ↑</th>
      <th style="text-align: center;">Missed-NOINFO ↓</th>
      <th style="text-align: center;">Missed-CONFLICTING ↓</th>
      <th style="text-align: center;">Over-SUPPORTS ↓</th>
      <th style="text-align: center;">Over-REFUTES ↓</th>
    </tr>
  </thead>
  <tbody>
    <!-- Qwen 3.5 9B -->
    <tr>
      <td rowspan="3" style="border-right: 1px solid #94a3b8;"><b>Qwen 3.5 9B</b></td>
      <td style="border-right: 1px solid #94a3b8;">Baseline</td>
      <td align="right">0.793</td>
      <td align="right">0.260</td>
      <td align="right">0.023</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.185</td>
      <td align="right">0.532</td>
      <td align="right">0.352</td>
      <td align="right">0.832</td>
      <td align="right">0.112</td>
      <td align="right">0.374</td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>UOC (Ours)</b></td>
      <td align="right"><b>0.816</b></td>
      <td align="right"><b>0.068</b></td>
      <td align="right"><b>0.006</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;"><b>0.063</b></td>
      <td align="right"><b>0.624</b></td>
      <td align="right"><b>0.055</b></td>
      <td align="right"><b>0.474</b></td>
      <td align="right"><b>0.051</b></td>
      <td align="right"><b>0.152</b></td>
    </tr>
    <tr style="background-color: #fef9c3;">
      <td style="color: #92400e; border-right: 1px solid #94a3b8;"><b>Δ</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.023</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.192</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.017</b></td>
      <td align="right" style="color: #16a34a; border-right: 1px solid #94a3b8;"><b>−0.122</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.092</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.297</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.358</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.061</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.222</b></td>
    </tr>
    <!-- GPT-OSS 20B -->
    <tr>
      <td rowspan="3" style="border-right: 1px solid #94a3b8;"><b>GPT-OSS 20B</b></td>
      <td style="border-right: 1px solid #94a3b8;">Baseline</td>
      <td align="right">0.693</td>
      <td align="right">0.625</td>
      <td align="right">0.121</td>
      <td align="right" style="border-right: 1px solid #94a3b8;">0.280</td>
      <td align="right">0.471</td>
      <td align="right">0.770</td>
      <td align="right">0.953</td>
      <td align="right">0.117</td>
      <td align="right">0.556</td>
    </tr>
    <tr>
      <td style="border-right: 1px solid #94a3b8;"><b>UOC (Ours)</b></td>
      <td align="right"><b>0.822</b></td>
      <td align="right"><b>0.084</b></td>
      <td align="right"><b>0.018</b></td>
      <td align="right" style="border-right: 1px solid #94a3b8;"><b>0.062</b></td>
      <td align="right"><b>0.627</b></td>
      <td align="right"><b>0.242</b></td>
      <td align="right"><b>0.647</b></td>
      <td align="right"><b>0.085</b></td>
      <td align="right"><b>0.233</b></td>
    </tr>
    <tr style="background-color: #fef9c3;">
      <td style="color: #92400e; border-right: 1px solid #94a3b8;"><b>Δ</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.129</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.541</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.103</b></td>
      <td align="right" style="color: #16a34a; border-right: 1px solid #94a3b8;"><b>−0.218</b></td>
      <td align="right" style="color: #16a34a;"><b>+0.156</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.528</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.306</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.032</b></td>
      <td align="right" style="color: #16a34a;"><b>−0.323</b></td>
    </tr>
  </tbody>
</table>

> **Missed-NOINFO** = predicts a verdict when gold is Not Enough Info &nbsp;·&nbsp; **Missed-CONFLICTING** = predicts a verdict when gold is Conflicting &nbsp;·&nbsp; **Over-SUPPORTS** = predicts SUPPORTS when gold is any non-SUPPORTS label &nbsp;·&nbsp; **Over-REFUTES** = predicts REFUTES when gold is any non-REFUTES label
