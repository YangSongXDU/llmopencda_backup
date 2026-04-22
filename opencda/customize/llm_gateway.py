# -*- coding: utf-8 -*-

import json
import socket
import urllib.request
import urllib.error


class LLMSemanticGateway(object):
    """负责向本地 llm_service 发起 HTTP 请求"""

    def __init__(self, endpoint, timeout=5.0, enabled=True):
        self.endpoint = endpoint
        self.timeout = timeout
        self.enabled = enabled

    def reason_overtake(self, payload):
        """
        向 llm_service 发送超车推理请求。

        Args:
            payload (dict): 符合 llm_service SceneRequest 结构的字典

        Returns:
            dict or None
        """
        if not self.enabled:
            print("[LLM_GATEWAY] disabled")
            return None

        try:
            data = json.dumps(payload).encode("utf-8")

            req = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8")

            result = json.loads(body)
            print("[LLM_GATEWAY] success decision=%s confidence=%s" % (
                result.get("decision"),
                result.get("confidence")
            ))
            return result

        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                err_body = "<no body>"
            print("[LLM_GATEWAY] HTTPError code=%s body=%s" % (e.code, err_body))
            return None

        except urllib.error.URLError as e:
            print("[LLM_GATEWAY] URLError reason=%s endpoint=%s" % (e.reason, self.endpoint))
            return None

        except socket.timeout:
            print("[LLM_GATEWAY] socket timeout endpoint=%s timeout=%.2f" % (
                self.endpoint, self.timeout
            ))
            return None

        except ValueError as e:
            print("[LLM_GATEWAY] JSON parse error: %s" % str(e))
            return None

        except Exception as e:
            print("[LLM_GATEWAY] unexpected error: %s" % str(e))
            return None