const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

const NIM_API_BASE = 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// Set to true to show <think> reasoning blocks
const SHOW_REASONING = true;

const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'meta/llama-3.1-8b-instruct',
  'gpt-4':         'meta/llama-3.1-70b-instruct',
  'gpt-4-turbo':   'meta/llama-3.1-405b-instruct',
  'gpt-4o':        'deepseek-ai/deepseek-v3-0324',
  'claude-3-opus': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'claude-3-sonnet':'mistralai/mistral-large-2-instruct',
  'gemini-pro':    'qwen/qwen3-235b-a22b-instruct',
  'glm-5':         'z-ai/glm5',
  'glm-4':         'z-ai/glm4.7'
};

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'OpenAI to NIM Proxy', reasoning: SHOW_REASONING });
});

app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  res.json({ object: 'list', data: models });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    const nimModel = MODEL_MAPPING[model] || 'meta/llama-3.1-70b-instruct';

    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.7,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          'Authorization': `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        responseType: stream ? 'stream' : 'json',
        timeout: 120000
      }
    );

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningBuffer = '';
      let reasoningSent = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;

          if (line.includes('[DONE]')) {
            res.write('data: [DONE]\n\n');
            return;
          }

          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;

            if (delta) {
              const reasoning = delta.reasoning_content;
              const content = delta.content;

              delete delta.reasoning_content;

              if (SHOW_REASONING) {
                if (reasoning) {
                  // Accumulate reasoning and wrap in <think> tags
                  if (!reasoningSent && reasoningBuffer === '') {
                    // First reasoning chunk — open the tag
                    delta.content = '<think>\n' + reasoning;
                  } else {
                    delta.content = reasoning;
                  }
                  reasoningBuffer += reasoning;
                } else if (content) {
                  if (reasoningBuffer && !reasoningSent) {
                    // First content chunk after reasoning — close the tag
                    delta.content = '\n</think>\n\n' + content;
                    reasoningSent = true;
                  } else {
                    delta.content = content;
                  }
                } else {
                  delta.content = '';
                }
              } else {
                // Reasoning hidden — only pass real content
                if (content !== undefined) {
                  delta.content = content;
                }
              }
            }

            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            // skip malformed chunks
          }
        });
      });

      response.data.on('end', () => res.end());
      response.data.on('error', () => res.end());

    } else {
      const choice = response.data.choices[0];
      let content = choice.message.content || '';

      if (SHOW_REASONING && choice.message.reasoning_content) {
        content = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + content;
      }

      res.json({
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: [{
          index: 0,
          message: {
            role: choice.message.role,
            content: content
          },
          finish_reason: choice.finish_reason
        }],
        usage: response.data.usage || {}
      });
    }

  } catch (error) {
    console.error('Error:', error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'proxy_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `${req.path} not found` } });
});

app.listen(PORT, () => {
  console.log(`NIM Proxy running on port ${PORT}`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ON' : 'OFF'}`);
});
